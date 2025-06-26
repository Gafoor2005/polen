// Main JavaScript for Pollen Classification System

document.addEventListener('DOMContentLoaded', function() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('fade-in');
        }, index * 100);
    });

    // Add pulse animation to important buttons
    const importantButtons = document.querySelectorAll('.btn-lg');
    importantButtons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.classList.add('pulse');
        });
        button.addEventListener('mouseleave', function() {
            this.classList.remove('pulse');
        });
    });

    // Smooth scrolling for internal links
    const internalLinks = document.querySelectorAll('a[href^="#"]');
    internalLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // File upload validation
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            const maxSize = 16 * 1024 * 1024; // 16MB
            const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];

            if (file) {
                // Check file size
                if (file.size > maxSize) {
                    alert('File size is too large. Please select a file smaller than 16MB.');
                    this.value = '';
                    return;
                }

                // Check file type
                if (!allowedTypes.includes(file.type)) {
                    alert('Invalid file type. Please select an image file (JPG, PNG, BMP, TIFF).');
                    this.value = '';
                    return;
                }

                // Update submit button text
                const submitBtn = document.getElementById('submitBtn');
                if (submitBtn) {
                    submitBtn.innerHTML = '<i class="fas fa-brain me-2"></i>Classify ' + file.name;
                }
            }
        });
    }

    // Form submission handling
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('file');
            if (!fileInput.files[0]) {
                e.preventDefault();
                alert('Please select an image file first.');
                return;
            }

            // Show loading state
            const submitBtn = document.getElementById('submitBtn');
            const loadingIndicator = document.getElementById('loadingIndicator');
            
            if (submitBtn) {
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
                submitBtn.disabled = true;
            }
            
            if (loadingIndicator) {
                loadingIndicator.style.display = 'block';
            }
        });
    }

    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert.classList.contains('show')) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    });

    // Add tooltips to elements with title attribute
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[title]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Progress bar animation for confidence scores
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const targetWidth = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.width = targetWidth;
        }, 500);
    });

    // Image preview enhancement
    const imagePreview = document.getElementById('imagePreview');
    if (imagePreview) {
        // Add drag and drop functionality
        imagePreview.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('border-success');
            this.style.backgroundColor = '#d4edda';
        });

        imagePreview.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('border-success');
            this.style.backgroundColor = '';
        });

        imagePreview.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('border-success');
            this.style.backgroundColor = '';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const fileInput = document.getElementById('file');
                if (fileInput) {
                    fileInput.files = files;
                    fileInput.dispatchEvent(new Event('change'));
                }
            }
        });

        // Click to upload
        imagePreview.addEventListener('click', function() {
            const fileInput = document.getElementById('file');
            if (fileInput) {
                fileInput.click();
            }
        });
    }
});

// Utility functions
function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    // Add to toast container or create one
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }

    toastContainer.appendChild(toast);

    // Show toast
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();

    // Remove toast after it's hidden
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

// API function for programmatic predictions
async function predictImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (result.success) {
            return {
                success: true,
                predicted_class: result.predicted_class,
                confidence: result.confidence
            };
        } else {
            return {
                success: false,
                error: result.error
            };
        }
    } catch (error) {
        return {
            success: false,
            error: 'Network error: ' + error.message
        };
    }
}

// Copy results to clipboard
function copyResults(predicted_class, confidence) {
    const text = `Pollen Classification Result:\nSpecies: ${predicted_class}\nConfidence: ${confidence}%\nGenerated by: Pollen Classification System`;
    
    navigator.clipboard.writeText(text).then(function() {
        showToast('Results copied to clipboard!', 'success');
    }).catch(function() {
        showToast('Failed to copy to clipboard', 'danger');
    });
}

// Share results (if Web Share API is supported)
function shareResults(predicted_class, confidence) {
    if (navigator.share) {
        navigator.share({
            title: 'Pollen Classification Result',
            text: `Identified pollen species: ${predicted_class} (${confidence}% confidence)`,
            url: window.location.href
        }).catch(console.error);
    } else {
        // Fallback to copying
        copyResults(predicted_class, confidence);
    }
}
