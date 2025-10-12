const fileUpload = document.getElementById('file-upload');

// Preview uploaded image
function showPreview(event) {
    const previewImg = document.getElementById('preview-image');
    const uploadLabel = document.getElementById('upload-label');
    const uploadIcon = document.getElementById('upload-icon');
    const resultImg = document.getElementById('result-image');

    // Clear previous result image if any
    if (resultImg) {
        resultImg.src = '#';
        resultImg.classList.add('hidden');
    }

    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.src = e.target.result;
            previewImg.classList.remove('hidden');
            uploadLabel.classList.add('hidden');
            uploadIcon.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        previewImg.classList.add('hidden');
        uploadLabel.classList.remove('hidden');
        uploadIcon.classList.remove('hidden');
    }
}

// Clear inputs, hide preview and spinner, restore upload label
function resetForm() {
    const fileUpload = document.getElementById('file-upload');
    const previewImg = document.getElementById('preview-image');
    const resultImg = document.getElementById('result-image');
    const message = document.getElementById('server-message');
    const uploadIcon = document.getElementById('upload-icon');
    const spinner = document.getElementById('spinner');
    const uploadLabel = document.getElementById('upload-label');

    const form = document.querySelector('form');
    if (!form) return;

    // Clear file input
    if (fileUpload) {
        fileUpload.value = '';
    }

    // Hide preview and clear src
    if (previewImg) {
        previewImg.classList.add('hidden');
        previewImg.src = '#';
    }
    
    // Reset upload label and icon
    if (uploadLabel) {
        uploadLabel.classList.remove('hidden');
    }
    if (uploadIcon) {
        uploadIcon.classList.remove('hidden');
    }

    // Reset result image
    if (resultImg) {
        resultImg.src = '#';
        resultImg.classList.add('hidden');
    }

    // Reset message
    if (message) {
        message.textContent = "Results will appear here after analysis";
    }

    // Hide spinner
    if (spinner) {
        spinner.classList.add('hidden');
    }
}

// Ensure file input change uses showPreview
if (fileUpload) {
    fileUpload.removeEventListener('change', showPreview); // Remove existing if any
    fileUpload.addEventListener('change', showPreview);
}

// Ajax for page reloading
// Immediately show the result without reloading the entire page
function submitForm(event) {
    event.preventDefault();
    let form = $('form')[0];
    let formData = new FormData(form);
    const spinner = $('#spinner');
    // We'll query the message element from the DOM after the server response

    spinner.removeClass('hidden');
    
    $.ajax({
        url: $(form).attr('action'),
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            console.log("Upload successful");
            // Replace the result section
            $('.result').html($(response).find('.result').html());

            // We received a response; hide the spinner
            spinner.addClass('hidden');

            // After replacing .result, query the new message element (if any)
            const newMessage = $('#message');
            newMessage.removeClass('hidden');
        },
        error: function(xhr, status, error) {
            console.error("Upload failed:", error);
            // Hide spinner on error and reveal message area if available
            $('#spinner').addClass('hidden');
            const newMessage = $('#message');
            if (newMessage.length) newMessage.removeClass('hidden');
        }
    });
}
