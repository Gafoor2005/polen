# Pollen Classification Web Application

A Flask-based web application for classifying Brazilian Savannah pollen species using deep learning.

## Features

- **Image Upload**: Easy drag-and-drop or click-to-upload interface
- **AI-Powered Classification**: CNN model trained on Brazilian Savannah pollen dataset
- **Real-time Results**: Instant classification with confidence scores
- **Responsive Design**: Works on desktop and mobile devices
- **Educational Information**: Detailed species information and system documentation

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Navigate to the application directory:**
   ```bash
   cd application
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are in place:**
   - The application looks for `final_pollen_classifier_model.h5` in the parent directory
   - Alternatively, place `best_pollen_model.h5` in the `../models/` directory
   - If you have a label encoder file, place it at `../processed_data/label_encoder.pkl`

### Running the Application

1. **Start the Flask development server:**
   ```bash
   python app.py
   ```

2. **Open your web browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Upload an image and get predictions!**

## Project Structure

```
application/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   ├── js/
│   │   └── main.js       # JavaScript functionality
│   └── uploads/          # Temporary upload directory
└── templates/
    ├── base.html         # Base template
    ├── index.html        # Home page
    ├── upload.html       # Upload page
    ├── result.html       # Results page
    └── about.html        # About page
```

## Usage

### Web Interface

1. **Home Page**: Overview of the system and navigation
2. **Upload Page**: 
   - Select or drag-and-drop a pollen microscopy image
   - Supported formats: JPG, PNG, JPEG, BMP, TIFF
   - Maximum file size: 16MB
3. **Results Page**: 
   - View predicted species with confidence score
   - Species information and details
   - Option to classify another image

### API Endpoint

The application also provides a REST API endpoint:

```bash
POST /api/predict
Content-Type: multipart/form-data
Body: file (image file)

Response:
{
  "predicted_class": "species_name",
  "confidence": 85.7,
  "success": true
}
```

## Supported Pollen Species

The model can identify the following Brazilian Savannah pollen species:

- Anadenanthera
- Arecaceae (Palm family)
- Arrabidaea
- Cecropia
- Chamaesyce
- Combretum
- Croton
- Cuphea
- Hymenaea
- Melastomataceae
- Mimosa
- Mouriri
- Piper
- Poaceae (Grass family)
- Protium
- Psidium
- Qualea
- Rubiaceae (Coffee family)
- Scoparia
- Sida
- Syagrus
- Tapirira
- Tibouchina

## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 128x128 pixels, RGB
- **Training Dataset**: Brazilian Savannah Pollen Dataset
- **Framework**: TensorFlow/Keras

## Browser Compatibility

- Chrome 60+
- Firefox 60+
- Safari 12+
- Edge 79+

## Troubleshooting

### Common Issues

1. **Model not found error:**
   - Ensure `final_pollen_classifier_model.h5` is in the parent directory
   - Check file permissions

2. **Memory errors:**
   - Reduce image size before upload
   - Ensure sufficient system RAM (4GB+ recommended)

3. **Slow predictions:**
   - Use GPU acceleration if available
   - Reduce image resolution

### Performance Tips

- Use images with good contrast and focus
- Center the pollen grain in the image
- Avoid images with multiple overlapping grains
- Optimal image size: 500x500 to 1000x1000 pixels

## Development

### Adding New Features

1. **Frontend changes:** Modify templates in `templates/` directory
2. **Styling:** Update `static/css/style.css`
3. **JavaScript:** Add functionality to `static/js/main.js`
4. **Backend:** Modify `app.py` for new routes or functionality

### Testing

Test the application with various image types and sizes to ensure robustness.

## Security Considerations

- File upload validation is implemented
- File size limits are enforced
- Temporary files are cleaned up after processing
- Consider adding authentication for production use

## License

This project is for educational and research purposes. Please refer to the main project license for detailed terms.

## Contact

For questions or issues, please refer to the main project documentation or contact the development team.
