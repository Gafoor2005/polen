# Pollen Dataset Structure

This directory contains microscopy images of Brazilian Savannah pollen species.

## Classes (23 total):

1. **anadenanthera** - 20 images (anadenanthera_16.jpg to anadenanthera_35.jpg)
2. **arecaceae** - 35 images (arecaceae_01.jpg to arecaceae_35.jpg)
3. **arrabidaea** - 35 images (arrabidaea_01.jpg to arrabidaea_35.jpg)
4. **cecropia** - 35 images (cecropia_01.jpg to cecropia_35.jpg)
5. **chamaesyce** - 20 images (chamaesyce_16.jpg to chamaesyce_35.jpg)
6. **combretum** - 20 images (combretum_16.jpg to combretum_35.jpg)
7. **croton** - 20 images (croton_16.jpg to croton_35.jpg)
8. **cuphea** - 20 images (cuphea_16.jpg to cuphea_35.jpg)
9. **hymenaea** - 20 images (hymenaea_16.jpg to hymenaea_35.jpg)
10. **melastomataceae** - 20 images (melastomataceae_16.jpg to melastomataceae_35.jpg)
11. **mimosa** - 20 images (mimosa_16.jpg to mimosa_35.jpg)
12. **mouriri** - 20 images (mouriri_16.jpg to mouriri_35.jpg)
13. **piper** - 20 images (piper_16.jpg to piper_35.jpg)
14. **poaceae** - 20 images (poaceae_16.jpg to poaceae_35.jpg)
15. **protium** - 20 images (protium_16.jpg to protium_35.jpg)
16. **psidium** - 20 images (psidium_16.jpg to psidium_35.jpg)
17. **qualea** - 20 images (qualea_16.jpg to qualea_35.jpg)
18. **rubiaceae** - 20 images (rubiaceae_16.jpg to rubiaceae_35.jpg)
19. **scoparia** - 20 images (scoparia_16.jpg to scoparia_35.jpg)
20. **senegalia** - 35 images (senegalia (1).jpg to senegalia (35).jpg)
21. **sida** - 20 images (sida_16.jpg to sida_35.jpg)
22. **syagrus** - 20 images (syagrus_16.jpg to syagrus_35.jpg)
23. **tapirira** - 20 images (tapirira_16.jpg to tapirira_35.jpg)
24. **tibouchina** - 20 images (tibouchina_16.jpg to tibouchina_35.jpg)
25. **urochloa** - 35 images (urochloa (1).jpg to urochloa (35).jpg)

## Total: ~550 images across 25 classes

## Naming Conventions:
- Most classes: `classname_number.jpg` (e.g., `anadenanthera_16.jpg`)
- Some classes: `classname (number).jpg` (e.g., `senegalia (1).jpg`)

## Image Properties:
- Format: JPEG
- Type: Microscopy images of pollen grains
- Source: Brazilian Savannah (Cerrado) species
- Quality: High-resolution microscopy images suitable for AI classification

## Usage:
These images are used to train the CNN model for automated pollen species identification.
The label extraction logic handles both naming conventions automatically.
