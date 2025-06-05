#!/usr/bin/env python3

import os
import yaml
import torch
import wandb
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score
import warnings
warnings.filterwarnings('ignore')

from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from models.tsn_paligemma_model import create_tsn_paligemma_model

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ModelTester:
    def __init__(self, config_path="config/config.yaml", model_path=None, test_data_path=None):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "test_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        self.base_model_id = self.config['model']['model_id']

        if model_path:
            self.trained_model_path = model_path
        else:
            # Prioritize best_model over final_model
            best_model_path = os.path.join(self.config['output']['output_dir'], "best_model")
            final_model_path = os.path.join(self.config['output']['output_dir'], "final_model")

            if os.path.exists(best_model_path):
                self.trained_model_path = best_model_path
                print("[BEST] Using best_model for testing")
            elif os.path.exists(final_model_path):
                self.trained_model_path = final_model_path
                print("[FILE] Using final_model for testing")
            else:
                raise FileNotFoundError("No trained model found in output directory")

        self.test_data_path = test_data_path
        print(f"[SUCCESS] Initialized tester")
        print(f"   Base model: {self.base_model_id}")
        print(f"   Trained model: {self.trained_model_path}")
        print(f"   Device: {self.device}")

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_test_data(self):
        """Load test dataset with max_val_samples limit"""
        if self.test_data_path:
            test_csv = self.test_data_path
        else:
            test_csv = 'processed_dataset/test.csv'
            if not os.path.exists(test_csv):
                test_csv = self.config['data']['val_csv']

        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Test dataset not found at: {test_csv}")

        df = pd.read_csv(test_csv)

        # Apply max_val_samples limit from config
        max_samples = self.config['data'].get('max_val_samples', -1)
        if max_samples > 0 and len(df) > max_samples:
            df = df.head(max_samples)
            print(f"[SUCCESS] Loaded {len(df)} test samples from {test_csv} (limited by max_val_samples: {max_samples})")
        else:
            print(f"[SUCCESS] Loaded {len(df)} test samples from {test_csv}")

        return df

    def load_base_model(self):
        print("[LOADING] Loading base PaliGemma model...")
        processor = PaliGemmaProcessor.from_pretrained(self.base_model_id)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.base_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print("[SUCCESS] Base model loaded")
        return model, processor

    def load_trained_model(self):
        print("[LOADING] Loading TSN-integrated trained model...")

        if not os.path.exists(self.trained_model_path):
            raise FileNotFoundError(f"Trained model not found at: {self.trained_model_path}")

        # Load processor
        try:
            processor = PaliGemmaProcessor.from_pretrained(self.trained_model_path)
            print("[SUCCESS] Loaded processor from trained model")
        except Exception:
            print("[WARNING]  Using base processor")
            processor = PaliGemmaProcessor.from_pretrained(self.base_model_id)

        # Load base model first
        print("[LOADING] Loading base PaliGemma model...")
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.base_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        # Apply LoRA adapter
        print("[LOADING] Applying LoRA adapter...")
        try:
            model = PeftModel.from_pretrained(base_model, self.trained_model_path)
            print("[SUCCESS] LoRA adapter applied successfully")
        except Exception as e:
            raise RuntimeError(f"Could not load LoRA adapter: {e}")

        # Wrap with TSN if config exists
        print("[LOADING] Checking for TSN integration...")
        try:
            # TSN model expects the LoRA-adapted model, not the base model
            tsn_model = create_tsn_paligemma_model(model, self.config)
            print("[SUCCESS] TSN wrapper applied successfully")
            return tsn_model, processor
        except Exception as e:
            print(f"[WARNING]  TSN integration failed, using LoRA model only: {e}")
            return model, processor

    def calculate_loss(self, model, processor, test_data, model_name):
        print(f"[LOADING] Calculating loss for {model_name}...")
        model.eval()
        total_loss = 0.0
        valid_samples = 0

        # Use all test data for loss calculation
        test_sample = test_data
        total_samples = len(test_sample)
        print(f"[DATA] Calculating loss on ALL {total_samples} samples")

        # Process in batches for memory efficiency
        batch_size = 4
        total_batches = (len(test_sample) + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in tqdm(range(total_batches), desc=f"Loss Calculation - {model_name}"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(test_sample))
                batch_df = test_sample.iloc[start_idx:end_idx]

                try:
                    batch_images = []
                    batch_texts = []
                    batch_labels = []

                    # Prepare batch data
                    for _, row in batch_df.iterrows():
                        img_path = os.path.normpath(row['image_path']).replace('\\', '/')
                        caption = row['caption']

                        if not os.path.exists(img_path):
                            continue

                        image = Image.open(img_path).convert('RGB')

                        # Try with suffix parameter first
                        try:
                            inputs = processor(
                                text=["<image>"],
                                images=[image],
                                suffix=[caption],
                                return_tensors="pt",
                                padding="longest"
                            )
                            batch_images.append(image)
                            batch_texts.append("<image>")

                        except (AttributeError, TypeError):
                            # Fallback to manual text construction
                            full_text = f"<image>{caption}"
                            batch_images.append(image)
                            batch_texts.append(full_text)

                    if not batch_images:
                        continue

                    # Process the batch
                    try:
                        # Try suffix method for the batch
                        batch_inputs = processor(
                            text=["<image>"] * len(batch_images),
                            images=batch_images,
                            suffix=[row['caption'] for _, row in batch_df.iterrows() if os.path.exists(os.path.normpath(row['image_path']).replace('\\', '/'))],
                            return_tensors="pt",
                            padding="longest"
                        ).to(self.device)
                    except (AttributeError, TypeError):
                        # Fallback to manual text construction for batch
                        batch_inputs = processor(
                            text=batch_texts,
                            images=batch_images,
                            return_tensors="pt",
                            padding="longest"
                        ).to(self.device)
                        # Add labels for loss calculation
                        batch_inputs["labels"] = batch_inputs["input_ids"].clone()

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        # Use underlying PaliGemma for loss calculation if TSN wrapper exists
                        if hasattr(model, 'paligemma'):
                            outputs = model.paligemma(**batch_inputs)
                        else:
                            outputs = model(**batch_inputs)

                        if hasattr(outputs, 'loss') and outputs.loss is not None:
                            total_loss += outputs.loss.item()
                            valid_samples += len(batch_images)

                    # Clear CUDA cache periodically
                    if batch_idx % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"[WARNING]  Batch {batch_idx} processing error: {str(e)[:50]}...")
                    continue

        avg_loss = total_loss / max(valid_samples, 1) if valid_samples > 0 else 0.0
        print(f"[SUCCESS] {model_name} - Average Loss: {avg_loss:.4f} (Valid samples: {valid_samples}/{total_samples})")
        return avg_loss, valid_samples

    def generate_captions(self, model, processor, test_data, model_name):
        print(f"[LOADING] Generating captions with {model_name}...")
        model.eval()
        generated_captions = []
        ground_truths = []

        # Process all test images
        test_sample = test_data
        total_samples = len(test_sample)
        print(f"[PROCESSING] Processing ALL {total_samples} test images")

        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for _, row in tqdm(test_sample.iterrows(), total=len(test_sample), desc=f"Captions - {model_name}"):
                img_path = os.path.normpath(row['image_path']).replace('\\', '/')
                caption = row['caption']

                try:
                    if not os.path.exists(img_path):
                        generated_captions.append("")
                        ground_truths.append(caption)
                        continue

                    image = Image.open(img_path).convert('RGB')

                    # Use proper PaliGemma prompt format for caption generation
                    prompt_text = "<image>describe this image"

                    try:
                        # Standard format with proper prompt
                        inputs = processor(
                            images=image,
                            text=prompt_text,
                            return_tensors="pt",
                            padding=False
                        ).to(self.device)
                    except Exception:
                        try:
                            # With explicit padding
                            inputs = processor(
                                images=image,
                                text=prompt_text,
                                return_tensors="pt",
                                padding=True,
                                max_length=512,
                                truncation=True
                            ).to(self.device)
                        except Exception:
                            # Fallback to CPU processing
                            inputs = processor(
                                images=image,
                                text=prompt_text,
                                return_tensors="pt",
                                padding=False
                            )
                            # Move to device after processing
                            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                    # Generate with token validation
                    vocab_size = len(processor.tokenizer)
                    pad_token_id = processor.tokenizer.pad_token_id
                    eos_token_id = processor.tokenizer.eos_token_id

                    # Validate token IDs to prevent CUDA errors
                    if pad_token_id is None or pad_token_id >= vocab_size:
                        pad_token_id = processor.tokenizer.eos_token_id
                    if eos_token_id is None or eos_token_id >= vocab_size:
                        eos_token_id = processor.tokenizer.eos_token_id

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        try:
                            # Use the underlying PaliGemma model for generation if TSN wrapper exists
                            if hasattr(model, 'paligemma'):
                                outputs = model.paligemma.generate(
                                    **inputs,
                                    max_new_tokens=50,
                                    num_beams=3,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.9,
                                    pad_token_id=pad_token_id,
                                    eos_token_id=eos_token_id,
                                    use_cache=True,
                                    output_scores=False,
                                    return_dict_in_generate=False
                                )
                            else:
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=50,
                                    num_beams=3,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.9,
                                    pad_token_id=pad_token_id,
                                    eos_token_id=eos_token_id,
                                    use_cache=True,
                                    output_scores=False,
                                    return_dict_in_generate=False
                                )
                        except Exception:
                            # Fallback generation
                            if hasattr(model, 'paligemma'):
                                outputs = model.paligemma.generate(
                                    input_ids=inputs['input_ids'],
                                    pixel_values=inputs['pixel_values'],
                                    max_new_tokens=30,
                                    do_sample=True,
                                    temperature=0.8,
                                    pad_token_id=pad_token_id,
                                    eos_token_id=eos_token_id
                                )
                            else:
                                outputs = model.generate(
                                    input_ids=inputs['input_ids'],
                                    pixel_values=inputs['pixel_values'],
                                    max_new_tokens=30,
                                    do_sample=True,
                                    temperature=0.8,
                                    pad_token_id=pad_token_id,
                                    eos_token_id=eos_token_id
                                )

                    # Decode safely
                    try:
                        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                        # Clean up the generated text
                        generated_caption = generated_text.replace("<image>", "").strip()
                        generated_caption = generated_caption.replace("describe this image", "").strip()

                        # Remove common prefixes/suffixes
                        if generated_caption.startswith("."):
                            generated_caption = generated_caption[1:].strip()
                        if generated_caption.startswith(":"):
                            generated_caption = generated_caption[1:].strip()
                        if generated_caption.startswith("caption:"):
                            generated_caption = generated_caption.replace("caption:", "").strip()
                        if generated_caption.startswith("description:"):
                            generated_caption = generated_caption.replace("description:", "").strip()

                        # Ensure meaningful content
                        if len(generated_caption.strip()) < 3:
                            generated_caption = ""

                    except Exception:
                        generated_caption = ""

                    generated_captions.append(generated_caption)
                    ground_truths.append(caption)

                except Exception as e:
                    print(f"[ERROR] Error generating caption: {str(e)[:100]}...")
                    generated_captions.append("")
                    ground_truths.append(caption)

                    # Clear CUDA cache on error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Count successful generations
        successful_gens = sum(1 for cap in generated_captions if len(cap.strip()) > 0)
        print(f"[SUCCESS] Generated {successful_gens}/{len(generated_captions)} successful captions from {total_samples} samples")
        return generated_captions, ground_truths

    def calculate_metrics(self, generated_captions, ground_truths):
        print("[LOADING] Calculating comprehensive evaluation metrics...")

        # Basic metrics
        total_samples = len(generated_captions)
        valid_generations = sum(1 for cap in generated_captions if len(cap.strip()) > 0)
        success_rate = valid_generations / total_samples if total_samples > 0 else 0
        avg_caption_length = np.mean([len(cap) for cap in generated_captions])

        # Initialize metric lists
        bleu_scores = []
        meteor_scores = []
        rouge_l_scores = []
        word_overlaps = []

        print("[DATA] Calculating BLEU, METEOR, ROUGE-L scores...")

        # Calculate individual scores
        smoothing = SmoothingFunction().method1
        for gen_cap, gt_cap in tqdm(zip(generated_captions, ground_truths), total=len(generated_captions), desc="Computing metrics"):
            if len(gt_cap.strip()) == 0 or len(gen_cap.strip()) == 0:
                bleu_scores.append(0.0)
                meteor_scores.append(0.0)
                rouge_l_scores.append(0.0)
                word_overlaps.append(0.0)
                continue

            # Tokenize
            gen_tokens = gen_cap.lower().split()
            gt_tokens = gt_cap.lower().split()

            # BLEU Score (1-4 grams average)
            try:
                bleu_1 = sentence_bleu([gt_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
                bleu_2 = sentence_bleu([gt_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
                bleu_3 = sentence_bleu([gt_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
                bleu_4 = sentence_bleu([gt_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                bleu_avg = (bleu_1 + bleu_2 + bleu_3 + bleu_4) / 4
                bleu_scores.append(bleu_avg)
            except:
                bleu_scores.append(0.0)

            # METEOR Score
            try:
                meteor = meteor_score([gt_cap.lower()], gen_cap.lower())
                meteor_scores.append(meteor)
            except:
                meteor_scores.append(0.0)

            # ROUGE-L Score
            try:
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge_scores = scorer.score(gt_cap.lower(), gen_cap.lower())
                rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
            except:
                rouge_l_scores.append(0.0)

            # Word Overlap
            gen_words = set(gen_tokens)
            gt_words = set(gt_tokens)
            if len(gt_words) > 0:
                overlap = len(gen_words.intersection(gt_words)) / len(gt_words)
            else:
                overlap = 0
            word_overlaps.append(overlap)

        # Calculate CIDEr and SPICE scores (corpus-level)
        print("[DATA] Calculating CIDEr and SPICE scores...")
        cider_score = self.calculate_cider_score(generated_captions, ground_truths)
        spice_score = self.calculate_spice_score(generated_captions, ground_truths)

        # Calculate BERTScore
        print("[DATA] Calculating BERTScore...")
        bert_precision, bert_recall, bert_f1 = self.calculate_bert_score(generated_captions, ground_truths)

        metrics = {
            'success_rate': success_rate,
            'avg_caption_length': avg_caption_length,
            'valid_generations': valid_generations,
            'total_samples': total_samples,
            'word_overlap': np.mean(word_overlaps) if word_overlaps else 0.0,
            'bleu_score': np.mean(bleu_scores) if bleu_scores else 0.0,
            'meteor_score': np.mean(meteor_scores) if meteor_scores else 0.0,
            'rouge_l_score': np.mean(rouge_l_scores) if rouge_l_scores else 0.0,
            'cider_score': cider_score,
            'spice_score': spice_score,
            'bert_precision': bert_precision,
            'bert_recall': bert_recall,
            'bert_f1': bert_f1
        }

        print(f"[SUCCESS] Comprehensive metrics calculated:")
        print(f"   Success Rate: {success_rate:.3f}")
        print(f"   BLEU Score: {metrics['bleu_score']:.3f}")
        print(f"   METEOR Score: {metrics['meteor_score']:.3f}")
        print(f"   ROUGE-L Score: {metrics['rouge_l_score']:.3f}")
        print(f"   CIDEr Score: {metrics['cider_score']:.3f}")
        print(f"   SPICE Score: {metrics['spice_score']:.3f}")
        print(f"   BERTScore F1: {metrics['bert_f1']:.3f}")

        return metrics

    def calculate_individual_metrics(self, generated_captions, ground_truths):
        """Calculate individual metrics for each sample"""
        individual_metrics = {
            'bleu_scores': [],
            'meteor_scores': [],
            'rouge_l_scores': [],
            'cider_scores': [],
            'word_overlaps': [],
            'caption_lengths': [],
            'quality_scores': []
        }

        smoothing = SmoothingFunction().method1

        for gen_cap, gt_cap in zip(generated_captions, ground_truths):
            # Caption length
            individual_metrics['caption_lengths'].append(len(gen_cap))

            # Quality score (basic check)
            individual_metrics['quality_scores'].append(1.0 if len(gen_cap.strip()) > 5 else 0.0)

            if len(gt_cap.strip()) == 0 or len(gen_cap.strip()) == 0:
                individual_metrics['bleu_scores'].append(0.0)
                individual_metrics['meteor_scores'].append(0.0)
                individual_metrics['rouge_l_scores'].append(0.0)
                individual_metrics['cider_scores'].append(0.0)
                individual_metrics['word_overlaps'].append(0.0)
                continue

            # Tokenize
            gen_tokens = gen_cap.lower().split()
            gt_tokens = gt_cap.lower().split()

            # BLEU Score
            try:
                bleu_1 = sentence_bleu([gt_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
                bleu_2 = sentence_bleu([gt_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
                bleu_3 = sentence_bleu([gt_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
                bleu_4 = sentence_bleu([gt_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                bleu_avg = (bleu_1 + bleu_2 + bleu_3 + bleu_4) / 4
                individual_metrics['bleu_scores'].append(bleu_avg)
            except:
                individual_metrics['bleu_scores'].append(0.0)

            # METEOR Score
            try:
                meteor = meteor_score([gt_cap.lower()], gen_cap.lower())
                individual_metrics['meteor_scores'].append(meteor)
            except:
                individual_metrics['meteor_scores'].append(0.0)

            # ROUGE-L Score
            try:
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge_scores = scorer.score(gt_cap.lower(), gen_cap.lower())
                individual_metrics['rouge_l_scores'].append(rouge_scores['rougeL'].fmeasure)
            except:
                individual_metrics['rouge_l_scores'].append(0.0)

            # CIDEr Score (individual)
            try:
                gts = {0: [gt_cap.strip()]}
                res = {0: [gen_cap.strip()]}
                cider_scorer = Cider()
                cider_score, _ = cider_scorer.compute_score(gts, res)
                individual_metrics['cider_scores'].append(float(cider_score))
            except:
                individual_metrics['cider_scores'].append(0.0)

            # Word Overlap
            gen_words = set(gen_tokens)
            gt_words = set(gt_tokens)
            if len(gt_words) > 0:
                overlap = len(gen_words.intersection(gt_words)) / len(gt_words)
            else:
                overlap = 0
            individual_metrics['word_overlaps'].append(overlap)

        return individual_metrics

    def calculate_cider_score(self, generated_captions, ground_truths):
        try:
            # Prepare data for CIDEr
            gts = {}
            res = {}
            for i, (gen_cap, gt_cap) in enumerate(zip(generated_captions, ground_truths)):
                gts[i] = [gt_cap.strip()]
                res[i] = [gen_cap.strip()]

            # Calculate CIDEr
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
            return float(cider_score)
        except Exception as e:
            print(f"[WARNING]  CIDEr calculation failed: {e}")
            return 0.0

    def calculate_spice_score(self, generated_captions, ground_truths):
        try:
            # Prepare data for SPICE
            gts = {}
            res = {}
            for i, (gen_cap, gt_cap) in enumerate(zip(generated_captions, ground_truths)):
                gts[i] = [gt_cap.strip()]
                res[i] = [gen_cap.strip()]

            # Calculate SPICE
            spice_scorer = Spice()
            spice_score, _ = spice_scorer.compute_score(gts, res)
            return float(spice_score)
        except Exception as e:
            print(f"[WARNING]  SPICE calculation failed: {e}")
            return 0.0

    def calculate_bert_score(self, generated_captions, ground_truths):
        try:
            # Calculate BERTScore
            P, R, F1 = bert_score(generated_captions, ground_truths, lang="en", verbose=False)
            return float(P.mean()), float(R.mean()), float(F1.mean())
        except Exception as e:
            print(f"[WARNING]  BERTScore calculation failed: {e}")
            return 0.0, 0.0, 0.0

    def save_comparison_results(self, base_captions, tsn_captions, ground_truths, test_data, base_metrics, tsn_metrics):
        print("[LOADING] Saving comprehensive comparison results...")

        results_df = test_data.copy()
        results_df['ground_truth'] = ground_truths
        results_df['base_model_output'] = base_captions
        results_df['tsn_model_output'] = tsn_captions

        # Remove caption column if it exists
        if 'caption' in results_df.columns:
            results_df = results_df.drop('caption', axis=1)

        # Calculate individual metrics for both models
        print("[LOADING] Calculating individual metrics for base model...")
        base_individual = self.calculate_individual_metrics(base_captions, ground_truths)

        print("[LOADING] Calculating individual metrics for TSN model...")
        tsn_individual = self.calculate_individual_metrics(tsn_captions, ground_truths)

        # Add all base model metrics
        results_df['base_bleu_score'] = base_individual['bleu_scores']
        results_df['base_meteor_score'] = base_individual['meteor_scores']
        results_df['base_rouge_l_score'] = base_individual['rouge_l_scores']
        results_df['base_cider_score'] = base_individual['cider_scores']
        results_df['base_word_overlap'] = base_individual['word_overlaps']
        results_df['base_caption_length'] = base_individual['caption_lengths']
        results_df['base_quality_score'] = base_individual['quality_scores']

        # Add all TSN model metrics
        results_df['tsn_bleu_score'] = tsn_individual['bleu_scores']
        results_df['tsn_meteor_score'] = tsn_individual['meteor_scores']
        results_df['tsn_rouge_l_score'] = tsn_individual['rouge_l_scores']
        results_df['tsn_cider_score'] = tsn_individual['cider_scores']
        results_df['tsn_word_overlap'] = tsn_individual['word_overlaps']
        results_df['tsn_caption_length'] = tsn_individual['caption_lengths']
        results_df['tsn_quality_score'] = tsn_individual['quality_scores']

        # Calculate improvements for each metric
        results_df['bleu_improvement'] = np.array(tsn_individual['bleu_scores']) - np.array(base_individual['bleu_scores'])
        results_df['meteor_improvement'] = np.array(tsn_individual['meteor_scores']) - np.array(base_individual['meteor_scores'])
        results_df['rouge_l_improvement'] = np.array(tsn_individual['rouge_l_scores']) - np.array(base_individual['rouge_l_scores'])
        results_df['cider_improvement'] = np.array(tsn_individual['cider_scores']) - np.array(base_individual['cider_scores'])
        results_df['word_overlap_improvement'] = np.array(tsn_individual['word_overlaps']) - np.array(base_individual['word_overlaps'])
        results_df['quality_improvement'] = np.array(tsn_individual['quality_scores']) - np.array(base_individual['quality_scores'])

        output_file = os.path.join(self.output_dir, "comprehensive_comparison_results.csv")
        results_df.to_csv(output_file, index=False)
        print(f"[SUCCESS] Comprehensive comparison results saved: {output_file}")
        return results_df

    def create_improvement_chart(self, base_metrics, tsn_metrics):
        """Create comprehensive improvement visualization chart"""
        print("[LOADING] Creating improvement visualization chart...")

        # Prepare data for visualization
        metrics_names = ['BLEU', 'METEOR', 'ROUGE-L', 'CIDEr', 'SPICE', 'BERTScore F1', 'Word Overlap', 'Success Rate']
        base_values = [
            base_metrics['bleu_score'],
            base_metrics['meteor_score'],
            base_metrics['rouge_l_score'],
            base_metrics['cider_score'],
            base_metrics['spice_score'],
            base_metrics['bert_f1'],
            base_metrics['word_overlap'],
            base_metrics['success_rate']
        ]
        tsn_values = [
            tsn_metrics['bleu_score'],
            tsn_metrics['meteor_score'],
            tsn_metrics['rouge_l_score'],
            tsn_metrics['cider_score'],
            tsn_metrics['spice_score'],
            tsn_metrics['bert_f1'],
            tsn_metrics['word_overlap'],
            tsn_metrics['success_rate']
        ]

        # Calculate improvement percentages
        improvement_percentages = []
        for base_val, tsn_val in zip(base_values, tsn_values):
            if base_val > 0:
                improvement = ((tsn_val - base_val) / base_val) * 100
            else:
                improvement = 0
            improvement_percentages.append(improvement)

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Base vs TSN Model Performance Comparison', fontsize=16, fontweight='bold')

        # Plot 1: Side-by-side comparison
        x = np.arange(len(metrics_names))
        width = 0.35

        bars1 = ax1.bar(x - width/2, base_values, width, label='Base Model', alpha=0.8, color='#ff7f0e')
        bars2 = ax1.bar(x + width/2, tsn_values, width, label='TSN Model', alpha=0.8, color='#2ca02c')

        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Base Model vs TSN Model Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: Improvement percentages
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in improvement_percentages]
        bars3 = ax2.bar(metrics_names, improvement_percentages, color=colors, alpha=0.8)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('TSN Model Improvement over Base Model')
        ax2.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Add percentage labels
        for bar, pct in zip(bars3, improvement_percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{pct:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

        # Plot 3: Radar chart for normalized metrics
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]

        # Normalize values to 0-1 scale for radar chart
        max_vals = [max(b, t) for b, t in zip(base_values, tsn_values)]
        base_normalized = [b/m if m > 0 else 0 for b, m in zip(base_values, max_vals)]
        tsn_normalized = [t/m if m > 0 else 0 for t, m in zip(tsn_values, max_vals)]

        base_normalized += base_normalized[:1]
        tsn_normalized += tsn_normalized[:1]

        ax3.plot(angles, base_normalized, 'o-', linewidth=2, label='Base Model', color='#ff7f0e')
        ax3.fill(angles, base_normalized, alpha=0.25, color='#ff7f0e')
        ax3.plot(angles, tsn_normalized, 'o-', linewidth=2, label='TSN Model', color='#2ca02c')
        ax3.fill(angles, tsn_normalized, alpha=0.25, color='#2ca02c')

        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(metrics_names)
        ax3.set_ylim(0, 1)
        ax3.set_title('Normalized Performance Radar Chart')
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Summary statistics
        ax4.axis('off')
        summary_text = f"""
        [DATA] PERFORMANCE SUMMARY

        [BEST] Best Improvements:
        • {metrics_names[np.argmax(improvement_percentages)]}: {max(improvement_percentages):.1f}%
        • {metrics_names[np.argsort(improvement_percentages)[-2]]}: {sorted(improvement_percentages)[-2]:.1f}%

        Worst Performance:
        • {metrics_names[np.argmin(improvement_percentages)]}: {min(improvement_percentages):.1f}%

        Overall Statistics:
        • Average Improvement: {np.mean(improvement_percentages):.1f}%
        • Positive Improvements: {sum(1 for x in improvement_percentages if x > 0)}/{len(improvement_percentages)}
        • Total Samples: {tsn_metrics['total_samples']}
        • TSN Success Rate: {tsn_metrics['success_rate']:.1%}
        """

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        chart_path = os.path.join(self.output_dir, "improvement_analysis_chart.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[SUCCESS] Improvement chart saved: {chart_path}")
        return chart_path

    def run_test(self):
        print("[START] Starting Base vs TSN model comparison test...")
        print("="*60)

        # Load test data
        test_data = self.load_test_data()

        # Load both models
        base_model, base_processor = self.load_base_model()
        tsn_model, tsn_processor = self.load_trained_model()

        # Calculate losses
        print("\nCALCULATING LOSSES")
        print("="*40)
        base_loss, _ = self.calculate_loss(base_model, base_processor, test_data, "Base Model")
        tsn_loss, _ = self.calculate_loss(tsn_model, tsn_processor, test_data, "TSN Model")

        # Generate captions
        print("\nGENERATING CAPTIONS")
        print("="*40)
        base_captions, ground_truths = self.generate_captions(base_model, base_processor, test_data, "Base Model")
        tsn_captions, _ = self.generate_captions(tsn_model, tsn_processor, test_data, "TSN Model")

        # Calculate metrics
        print("\nCALCULATING METRICS")
        print("="*40)
        base_metrics = self.calculate_metrics(base_captions, ground_truths)
        tsn_metrics = self.calculate_metrics(tsn_captions, ground_truths)

        # Initialize WandB
        print("\nINITIALIZING WANDB")
        print("="*40)
        os.environ["WANDB_DIR"] = "."
        test_run_name = self.config['wandb'].get('test_run_name', 'base-vs-tsn-comparison')
        wandb.init(
            project=self.config['wandb']['project'] + "-test",
            name=test_run_name,
            config=self.config,
            dir=".",
            save_code=True,
            tags=["test", "comparison", "base-vs-tsn", "paligemma"]
        )

        # Log comprehensive comparison metrics to WandB
        wandb_metrics = {
            # Base model metrics
            "base_model/loss": float(base_loss),
            "base_model/success_rate": float(base_metrics['success_rate']),
            "base_model/word_overlap": float(base_metrics['word_overlap']),
            "base_model/bleu_score": float(base_metrics['bleu_score']),
            "base_model/meteor_score": float(base_metrics['meteor_score']),
            "base_model/rouge_l_score": float(base_metrics['rouge_l_score']),
            "base_model/cider_score": float(base_metrics['cider_score']),
            "base_model/spice_score": float(base_metrics['spice_score']),
            "base_model/bert_f1": float(base_metrics['bert_f1']),
            "base_model/bert_precision": float(base_metrics['bert_precision']),
            "base_model/bert_recall": float(base_metrics['bert_recall']),
            "base_model/avg_caption_length": float(base_metrics['avg_caption_length']),
            "base_model/valid_generations": int(base_metrics['valid_generations']),
            "base_model/total_samples": int(base_metrics['total_samples']),

            # TSN model metrics
            "tsn_model/loss": float(tsn_loss),
            "tsn_model/success_rate": float(tsn_metrics['success_rate']),
            "tsn_model/word_overlap": float(tsn_metrics['word_overlap']),
            "tsn_model/bleu_score": float(tsn_metrics['bleu_score']),
            "tsn_model/meteor_score": float(tsn_metrics['meteor_score']),
            "tsn_model/rouge_l_score": float(tsn_metrics['rouge_l_score']),
            "tsn_model/cider_score": float(tsn_metrics['cider_score']),
            "tsn_model/spice_score": float(tsn_metrics['spice_score']),
            "tsn_model/bert_f1": float(tsn_metrics['bert_f1']),
            "tsn_model/bert_precision": float(tsn_metrics['bert_precision']),
            "tsn_model/bert_recall": float(tsn_metrics['bert_recall']),
            "tsn_model/avg_caption_length": float(tsn_metrics['avg_caption_length']),
            "tsn_model/valid_generations": int(tsn_metrics['valid_generations']),
            "tsn_model/total_samples": int(tsn_metrics['total_samples']),

            # Comparison metrics - Loss
            "comparison/loss_improvement": float(base_loss - tsn_loss),
            "comparison/loss_improvement_percent": float(((base_loss - tsn_loss) / base_loss * 100) if base_loss > 0 else 0),

            # Comparison metrics - NLP Scores
            "comparison/bleu_improvement": float(tsn_metrics['bleu_score'] - base_metrics['bleu_score']),
            "comparison/bleu_improvement_percent": float(((tsn_metrics['bleu_score'] - base_metrics['bleu_score']) / base_metrics['bleu_score'] * 100) if base_metrics['bleu_score'] > 0 else 0),
            "comparison/meteor_improvement": float(tsn_metrics['meteor_score'] - base_metrics['meteor_score']),
            "comparison/meteor_improvement_percent": float(((tsn_metrics['meteor_score'] - base_metrics['meteor_score']) / base_metrics['meteor_score'] * 100) if base_metrics['meteor_score'] > 0 else 0),
            "comparison/rouge_l_improvement": float(tsn_metrics['rouge_l_score'] - base_metrics['rouge_l_score']),
            "comparison/rouge_l_improvement_percent": float(((tsn_metrics['rouge_l_score'] - base_metrics['rouge_l_score']) / base_metrics['rouge_l_score'] * 100) if base_metrics['rouge_l_score'] > 0 else 0),
            "comparison/cider_improvement": float(tsn_metrics['cider_score'] - base_metrics['cider_score']),
            "comparison/cider_improvement_percent": float(((tsn_metrics['cider_score'] - base_metrics['cider_score']) / base_metrics['cider_score'] * 100) if base_metrics['cider_score'] > 0 else 0),
            "comparison/spice_improvement": float(tsn_metrics['spice_score'] - base_metrics['spice_score']),
            "comparison/spice_improvement_percent": float(((tsn_metrics['spice_score'] - base_metrics['spice_score']) / base_metrics['spice_score'] * 100) if base_metrics['spice_score'] > 0 else 0),
            "comparison/bert_f1_improvement": float(tsn_metrics['bert_f1'] - base_metrics['bert_f1']),
            "comparison/bert_f1_improvement_percent": float(((tsn_metrics['bert_f1'] - base_metrics['bert_f1']) / base_metrics['bert_f1'] * 100) if base_metrics['bert_f1'] > 0 else 0),

            # Comparison metrics - Basic
            "comparison/success_rate_improvement": float(tsn_metrics['success_rate'] - base_metrics['success_rate']),
            "comparison/success_rate_improvement_percent": float(((tsn_metrics['success_rate'] - base_metrics['success_rate']) / base_metrics['success_rate'] * 100) if base_metrics['success_rate'] > 0 else 0),
            "comparison/word_overlap_improvement": float(tsn_metrics['word_overlap'] - base_metrics['word_overlap']),
            "comparison/word_overlap_improvement_percent": float(((tsn_metrics['word_overlap'] - base_metrics['word_overlap']) / base_metrics['word_overlap'] * 100) if base_metrics['word_overlap'] > 0 else 0),

            # Test info
            "test/total_samples_tested": int(len(test_data)),
            "test/model_path": str(self.trained_model_path),
            "test/completed": True
        }

        wandb.log(wandb_metrics)

        # Create comparison table for WandB
        sample_table_data = []
        for i, (base_cap, tsn_cap, gt) in enumerate(zip(base_captions[:20], tsn_captions[:20], ground_truths[:20])):
            sample_table_data.append([
                i + 1,
                gt[:60] + "..." if len(gt) > 60 else gt,
                base_cap[:60] + "..." if len(base_cap) > 60 else base_cap,
                tsn_cap[:60] + "..." if len(tsn_cap) > 60 else tsn_cap,
                len(base_cap),
                len(tsn_cap),
                1.0 if len(base_cap.strip()) > 5 else 0.0,
                1.0 if len(tsn_cap.strip()) > 5 else 0.0
            ])

        sample_table = wandb.Table(
            columns=["Sample_ID", "Ground_Truth", "Base_Model", "TSN_Model", "Base_Length", "TSN_Length", "Base_Quality", "TSN_Quality"],
            data=sample_table_data
        )
        wandb.log({"comparison_samples": sample_table})

        # Save results
        print("\nSAVING RESULTS")
        print("="*40)
        results_df = self.save_comparison_results(base_captions, tsn_captions, ground_truths, test_data, base_metrics, tsn_metrics)

        # Create and save improvement chart
        print("\nCREATING IMPROVEMENT VISUALIZATION")
        print("="*40)
        chart_path = self.create_improvement_chart(base_metrics, tsn_metrics)
        if chart_path:
            wandb.log({"improvement_analysis_chart": wandb.Image(chart_path)})

        # Print comprehensive summary
        print("\n" + "="*80)
        print("COMPREHENSIVE BASE vs TSN MODEL COMPARISON RESULTS")
        print("="*80)
        print(f"Total samples tested: {len(test_data)}")

        print("\nLoss Comparison:")
        print(f"   Base Model Loss: {base_loss:.4f}")
        print(f"   TSN Model Loss:  {tsn_loss:.4f}")
        print(f"   Improvement:     {base_loss - tsn_loss:+.4f} ({((base_loss - tsn_loss) / base_loss * 100) if base_loss > 0 else 0:+.2f}%)")

        print("\nNLP Evaluation Metrics Comparison:")
        print(f"   BLEU Score:    {base_metrics['bleu_score']:.3f} → {tsn_metrics['bleu_score']:.3f} ({((tsn_metrics['bleu_score'] - base_metrics['bleu_score']) / base_metrics['bleu_score'] * 100) if base_metrics['bleu_score'] > 0 else 0:+.2f}%)")
        print(f"   METEOR Score:  {base_metrics['meteor_score']:.3f} → {tsn_metrics['meteor_score']:.3f} ({((tsn_metrics['meteor_score'] - base_metrics['meteor_score']) / base_metrics['meteor_score'] * 100) if base_metrics['meteor_score'] > 0 else 0:+.2f}%)")
        print(f"   ROUGE-L Score: {base_metrics['rouge_l_score']:.3f} → {tsn_metrics['rouge_l_score']:.3f} ({((tsn_metrics['rouge_l_score'] - base_metrics['rouge_l_score']) / base_metrics['rouge_l_score'] * 100) if base_metrics['rouge_l_score'] > 0 else 0:+.2f}%)")
        print(f"   CIDEr Score:   {base_metrics['cider_score']:.3f} → {tsn_metrics['cider_score']:.3f} ({((tsn_metrics['cider_score'] - base_metrics['cider_score']) / base_metrics['cider_score'] * 100) if base_metrics['cider_score'] > 0 else 0:+.2f}%)")
        print(f"   SPICE Score:   {base_metrics['spice_score']:.3f} → {tsn_metrics['spice_score']:.3f} ({((tsn_metrics['spice_score'] - base_metrics['spice_score']) / base_metrics['spice_score'] * 100) if base_metrics['spice_score'] > 0 else 0:+.2f}%)")
        print(f"   BERTScore F1:  {base_metrics['bert_f1']:.3f} → {tsn_metrics['bert_f1']:.3f} ({((tsn_metrics['bert_f1'] - base_metrics['bert_f1']) / base_metrics['bert_f1'] * 100) if base_metrics['bert_f1'] > 0 else 0:+.2f}%)")

        print("\nBasic Performance Comparison:")
        print(f"   Success Rate:      {base_metrics['success_rate']:.3f} → {tsn_metrics['success_rate']:.3f} ({((tsn_metrics['success_rate'] - base_metrics['success_rate']) / base_metrics['success_rate'] * 100) if base_metrics['success_rate'] > 0 else 0:+.2f}%)")
        print(f"   Word Overlap:      {base_metrics['word_overlap']:.3f} → {tsn_metrics['word_overlap']:.3f} ({((tsn_metrics['word_overlap'] - base_metrics['word_overlap']) / base_metrics['word_overlap'] * 100) if base_metrics['word_overlap'] > 0 else 0:+.2f}%)")
        print(f"   Avg Caption Length: {base_metrics['avg_caption_length']:.1f} → {tsn_metrics['avg_caption_length']:.1f}")
        print(f"   Valid Generations:  {base_metrics['valid_generations']}/{base_metrics['total_samples']} → {tsn_metrics['valid_generations']}/{tsn_metrics['total_samples']}")

        print(f"\nResults saved to: {self.output_dir}/")
        print("="*80)

        wandb.finish()
        return results_df

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PaliGemma Model Test & Evaluation")

    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to trained model directory")
    parser.add_argument("--test-data", type=str, default=None,
                       help="Path to test dataset CSV file")
    parser.add_argument("--output-dir", type=str, default="test_outputs",
                       help="Directory to save test results")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    print("🧪 Base vs TSN-PaliGemma Model Comparison Test")
    print("="*50)

    tester = ModelTester(
        config_path=args.config,
        model_path=args.model_path,
        test_data_path=args.test_data
    )

    if args.output_dir != "test_outputs":
        tester.output_dir = args.output_dir
        os.makedirs(tester.output_dir, exist_ok=True)

    try:
        results_df = tester.run_test()
        print("\n[SUCCESS] Test completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
