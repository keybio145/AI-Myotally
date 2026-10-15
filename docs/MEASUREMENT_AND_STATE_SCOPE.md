# Measurements and saved decision scope

The supplied measurement table contains fiber types obtained by image-level calibration with the saved operator labels; values used for Figure 4 and Representative Results section 4. Eight decision metadata columns were restored from the same preserved per-section calibrated signals, fill/artifact features, thresholds, minimum fill fractions, positive overrides and explicitly negative labels using the original v1.0.0 classifier. Applied manual overrides have High confidence and the Manual override flag. The original 27 parsed field values and their precision are preserved.

When preparing the final measurement table, saved type labels on 2 objects that are not in the final table were not applied.

Reanalyzing the raw images interactively in the application may not reproduce identical measurements or decisions. Saved state is an endpoint snapshot and selected stage masks, not a chronological record of individual editing actions. Image-level calibration uses labels from the current image; a session-level profile is memory-only. This archive supplies prepared calibrated output and downstream scripts; it does not replay segmentation, RF training or interactive calibration from raw images.

Mask TIFFs retain original positive Mask_ID labels and zero background. Join the image name and label value to Image and Mask_ID in the measurement table to obtain Fiber_ID. Fiber_ID may be zero, so it is not used directly as a TIFF label. Every final label has exactly one measurement row and vice versa.

Old 3 and Young 1 include sanitized editor states, post-editing reference masks, raw segmentation and filter destination arrays, and geometry-reject previews. Original RF models are not distributed. The Figure 3 script redraws stages from these saved arrays and final measurement types. Display-crop coordinates are presentation choices and are distinct from validation ROI identifiers; no blinded validation ROI-to-section key is supplied.
