# Training Improvement Plan
**Based on Model Comparison Analysis - November 13, 2025**

## 🎯 Current Performance
- **Model 1 (1024×768):** F1: 0.6791, IoU: 0.5364, Edge IoU: 0.0440
- **Model 2 (1024×1024):** F1: 0.7078, IoU: 0.5681, Edge IoU: 0.0549
- **Winner:** Model 2 (+4.2% F1, +24.8% Edge IoU)

## ✅ What's Working Well
1. **Background separation is EXCELLENT** - No false positives on background
2. **Resolution 1024×1024** - Clear winner over 768×1024
3. **General stroke detection** - High recall (catching most strokes)

---

## ❌ Critical Issues to Fix (Visual Analysis)

### Issue 1: Gaps in Long Straight Lines
**Problem:** Long straight lines have breaks/discontinuities
**Why:** Model predicts stroke presence independently per-pixel without line continuity constraint
**Solutions:**
- [ ] Add **morphological closing** in post-processing (close small gaps)
- [ ] Implement **skeleton refinement** (connect nearby line segments)
- [ ] Add **line continuity loss** during training (penalize gaps in predicted strokes)
- [ ] Try **higher resolution** (1280×1280 or 1536×1536) for better sampling

**Priority:** HIGH - Affects line quality significantly

---

### Issue 2: Lines Not Staying Straight
**Problem:** Predicted lines are wobbly/jagged instead of smooth
**Why:** Pixel-level prediction without geometric constraints
**Solutions:**
- [ ] **Post-processing:** Apply Hough line detection + straighten detected lines
- [ ] **Training augmentation:** Add more straight line examples to dataset
- [ ] **Edge-preserving loss:** Penalize jagged boundaries (add sobel/canny edge loss)
- [ ] **Higher resolution:** 1280×1280 may help with smoother edges

**Priority:** MEDIUM-HIGH - Affects professional appearance

---

### Issue 3: Small Letters Grouped as Big Blobs
**Problem:** Individual characters merge into single large strokes
**Root Cause:** Resolution insufficient to separate close features (NOT mask quality - masks verified correct)
**Visual Evidence:** "My Name is Nick" - letters blur together despite clean masks
**Solutions:**
- [ ] **CRITICAL: Increase resolution to 1536×1536**
  - Small text needs 10-15 pixels minimum per letter for separation
  - Current 1024×1024 gives only 5-8 pixels for small text
  - Higher resolution will preserve gaps between letters
- [ ] **Verify it's resolution, not masks:** ✅ CONFIRMED - Masks are properly labeled and not merging
- [ ] Add **multi-scale feature pyramid** in model architecture
- [ ] Use **atrous/dilated convolutions** for better receptive field without resolution loss

**Priority:** CRITICAL - Core functionality for handwriting

---

### Issue 4: Squiggly/Curved Lines Not Picked Up
**Problem:** Complex curves, loops, or irregular strokes are missed
**Root Cause:** Unknown - need to investigate (dataset, threshold, or architecture)
**Masks Status:** ✅ NEED TO VERIFY - Check if squiggly lines are properly labeled in ground truth
**Solutions:**
- [ ] **First: Verify mask quality** - Check if training masks include squiggly lines properly
- [ ] **If masks good:** Add more curved/squiggly line examples to dataset
- [ ] **Try lower prediction threshold:** May be detecting but below confidence cutoff (currently 0.5)
- [ ] **Higher resolution helps:** 1536×1536 captures curve details better
- [ ] Add **curvature-aware augmentation** (elastic deformations during training)

**Priority:** HIGH - Handwriting has many curves

---

### Issue 5: Close Lines Get Merged
**Problem:** Nearby parallel strokes detected as single wide stroke
**Root Cause:** Resolution too low to capture gaps between strokes (NOT mask merging issue)
**Visual Evidence:** Mathematical symbols, closely-spaced text
**Masks Status:** ✅ VERIFIED - All masks properly labeled, none are merging in ground truth
**Solutions:**
- [ ] **CRITICAL: Increase resolution to 1536×1536 minimum**
  - Need 3-5 pixel gap between strokes to separate them
  - Current 1024×1024 only gives 1-2 pixel gap
  - Higher resolution = better gap preservation
- [ ] **Post-processing:** Add watershed algorithm to split any remaining merged regions
- [ ] **Architecture:** Try HRNet (maintains high resolution throughout network)

**Priority:** CRITICAL - Essential for dense handwriting

---

## 📋 Action Items for Next Training Run

### Immediate (Next Run)
1. **Increase resolution to 1536×1536**
   ```bash
   python train_segmentation.py \
       --img-height 1536 \
       --img-width 1536 \
       --batch-size 2 \
       --lr 1e-4 \
       --epochs 150 \
       --use-amp \
       --output-dir models-high-res
   ```
   - Expected: +10-15% improvement in small text separation
   - Trade-off: 2.25x slower training, ~16GB GPU memory needed

2. **Add edge-aware loss component**
   - Implement Sobel/Canny edge detection on predictions
   - Add edge loss term: `total_loss = dice + focal + edge_loss`
   - Should fix wobbly lines and improve Edge IoU

3. **Verify mask quality for problem cases**
   - Check masks for squiggly lines (are they properly labeled?)
   - Check masks for close strokes (are they separated or merged?)
   - Re-label if needed

### Short-term (Within Week)
4. **Augment dataset with specific problem cases:**
   - Collect 20-30 images with small text
   - Collect 20-30 images with curved/squiggly lines  
   - Collect 20-30 images with closely-spaced strokes
   - Label and add to training set

5. **Implement post-processing pipeline:**
   ```python
   def refine_predictions(mask):
       # Close small gaps in lines
       mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_size=3)
       
       # Connect nearby line segments
       mask = connect_line_segments(mask, max_gap=5)
       
       # Straighten detected lines
       mask = straighten_lines(mask, deviation_threshold=2)
       
       # Separate merged strokes
       mask = watershed_separation(mask)
       
       return mask
   ```

6. **Try different backbone architecture:**
   - Current: MobileNetV3 (fast but limited capacity)
   - Try: ResNet50 or EfficientNet-B3 (better feature extraction)
   - Expected: +5-8% F1 improvement

### Medium-term (Next 2 Weeks)
7. **Implement multi-scale training:**
   - Train on mix of resolutions: 1024, 1280, 1536, 2048
   - Model learns to handle different detail levels
   - Better generalization

8. **Add instance segmentation branch:**
   - Detect each stroke as separate instance
   - Prevents merging of close lines
   - More complex but much better quality

9. **Fine-tune on problem categories:**
   - Create separate dataset splits: small_text, curves, close_strokes
   - Train specialized models or fine-tune on each
   - Ensemble predictions

### Long-term (Future Improvements)
10. **Collect more diverse training data:**
    - Current: 184 training images (small dataset)
    - Target: 500-1000 images
    - Include all edge cases identified

11. **Implement active learning:**
    - Run model on unlabeled images
    - Find images where model struggles (low confidence)
    - Prioritize labeling those cases

12. **Try transformer-based architecture:**
    - SegFormer, Mask2Former, or SAM fine-tuning
    - Better at long-range dependencies (straight lines)
    - May handle small details better

---

## 🎯 Expected Improvements by Priority

| Issue | Current Impact | Priority | Expected Fix | Improvement |
|-------|---------------|----------|--------------|-------------|
| Small letters merged | CRITICAL | P0 | 1536×1536 resolution | +15-20% F1 |
| Close lines merged | CRITICAL | P0 | 1536×1536 + watershed | +10-15% F1 |
| Gaps in lines | HIGH | P1 | Morphological closing | +5-8% F1 |
| Squiggly lines missed | HIGH | P1 | Dataset augmentation | +8-12% F1 |
| Lines not straight | MEDIUM | P2 | Edge loss + post-proc | +3-5% F1 |
| Low Edge IoU | MEDIUM | P2 | Edge-aware loss | Edge IoU: 0.05→0.15 |

**Target Performance After Fixes:**
- F1: 0.85-0.90 (currently 0.71)
- IoU: 0.75-0.80 (currently 0.57)
- Edge IoU: 0.15-0.20 (currently 0.05)

---

## 🚀 Recommended Next Training Config

**Based on Comparison Summary Analysis:**

The comparison showed Model 2 (test F1: 0.7078) beat Model 1 (test F1: 0.6791), BUT the recommendations are confusing because they suggest reverting to Model 1 settings. Here's what ACTUALLY worked:

### ✅ What to KEEP from Model 2 (Winner)
- ✅ **Resolution: 1024×1024** - Edge IoU improved 24.8%! This is THE critical factor
- ✅ **AMP: Enabled** - Despite recommendation saying "disabled better," Model 2 WON with AMP on
- ✅ **Longer training: 150+ epochs** - Model 2 converged better

### ❌ What to CHANGE from Model 2
- ❌ **Batch size: 4 → 2** - Comparison correctly identified batch 2 gives better results
- ❌ **Learning rate: 1e-4 → 2e-4** - Higher LR from Model 1 performed better
- ❌ **Patience: 20 → 15** - Shorter patience is sufficient

### 🎯 OPTIMAL CONFIG (Hybrid Approach)
Take Model 2's WINNER features + Model 1's better hyperparameters + scale up resolution:

```bash
# Ultra-high resolution training with optimal hyperparameters
python train_segmentation.py \
    --img-height 1536 \
    --img-width 1536 \
    --batch-size 2 \
    --lr 2e-4 \
    --epochs 150 \
    --patience 15 \
    --use-amp \
    --output-dir models-ultra-high-res \
    --gradient-clip-max-norm 1.0
```

**Why this config:**
- **1536×1536:** Solves small text merging + close stroke separation (CRITICAL fix)
- **Batch 2:** Proven better than 4 in comparison (more gradient updates)
- **LR 2e-4:** Model 1's learning rate performed better than Model 2's 1e-4
- **AMP enabled:** Model 2 won WITH AMP despite recommendations saying otherwise
- **150 epochs:** Model 2 showed improvements up to epoch 136, needs full runway
- **Patience 15:** Model 1's patience worked well, no need for 20

**Expected results:**
- Training time: ~90-100 minutes (2.25x more pixels than 1024×1024)
- GPU memory: 14-16GB (may need batch_size=1 on 10GB RTX 3080)
- F1 improvement: **0.71 → 0.85-0.90** (+20-27% gain)
- Edge IoU: **0.055 → 0.15+** (3x better boundaries)

**If GPU memory issues (OOM errors):**
```bash
# Fallback: 1280×1280 resolution (60% more pixels, less memory)
python train_segmentation.py \
    --img-height 1280 \
    --img-width 1280 \
    --batch-size 2 \
    --lr 2e-4 \
    --epochs 150 \
    --patience 15 \
    --use-amp \
    --output-dir models-high-res-1280
```

Or use gradient accumulation:
```bash
# 1536×1536 with batch_size=1 + gradient accumulation to simulate batch_size=2
python train_segmentation.py \
    --img-height 1536 \
    --img-width 1536 \
    --batch-size 1 \
    --accumulation-steps 2 \
    --lr 2e-4 \
    --epochs 150 \
    --patience 15 \
    --use-amp \
    --output-dir models-ultra-high-res
```

---

## 📊 Success Metrics

Track these in next comparison:
- [ ] Small text F1 score (isolated metric for letter-heavy images)
- [ ] Close stroke separation accuracy (% of strokes correctly separated)
- [ ] Line straightness score (deviation from ideal straight line)
- [ ] Gap detection rate (% of lines without breaks)
- [ ] Curve detection F1 (squiggly vs straight line performance)

---

## 🔄 Iterative Improvement Process

1. **Train with 1536×1536** → Measure improvement
2. **Add edge loss** → Measure Edge IoU improvement  
3. **Implement post-processing** → Visual quality check
4. **Collect problem cases** → Retrain with augmented data
5. **Compare again** → Quantify gains
6. **Repeat** until target metrics achieved

---

## 📝 Notes
- **GPU Memory:** 1536×1536 may require 14-16GB VRAM (RTX 3080 10GB might need batch_size=1 or gradient accumulation)
- **Alternative:** Train at 1280×1280 first (50% more pixels, less memory) then scale to 1536
- **Dataset quality matters:** Fix/verify masks for problem cases before increasing resolution
- **Keep Model 2 config as baseline:** It's proven superior (1024×1024, batch 4, AMP, LR 1e-4)

---

**Last Updated:** November 13, 2025  
**Based On:** Model 1 vs Model 2 comparison (6 test images)  
**Next Review:** After 1536×1536 training completes
