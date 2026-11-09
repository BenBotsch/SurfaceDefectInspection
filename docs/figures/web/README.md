# Figures for web pages

The README figures from `docs/figures/`, prepared for use on other web pages. `sdi figures --web`
and `sdi gradcam --web` write them from the same results.

- Charts (`confusion`, `per_class_f1`, `training`, `robustness`) are SVG and stay sharp at any
  width.
- Photographs (`samples`, `errors`, `gradcam`) are WebP, 1600 pixels wide, which is sharp on
  high-density screens in a column of up to 800 CSS pixels.
- The background is transparent. Use `<name>.<ext>` on light pages and `<name>-dark.<ext>` on dark
  pages, because the text colour differs.
- Each figure is at most 6.4 inches wide, so its text is about 15 pixels high in a column of 800
  CSS pixels. Grad-CAM wraps into two rows and the robustness panels into a 3 x 2 grid.

Scale the images with the column, for example with `width: 100%; height: auto`.

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="robustness-dark.svg">
  <img src="robustness.svg" alt="Test accuracy of three models under five image corruptions"
       style="width: 100%; height: auto">
</picture>
```

## Attribution

`samples`, `errors` and `gradcam` contain images from NEU-DET and need a credit wherever they are
shown, for example:

> Images: NEU surface defect database (K. Song and Y. Yan, 2013), cleaned release
> [KeenForgeAI/NEU-DET-corrected](https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected),
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Downsized, Grad-CAM overlays added.

The charts contain no dataset images.
