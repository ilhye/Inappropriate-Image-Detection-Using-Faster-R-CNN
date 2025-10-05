// Preview uploaded image
function showPreview(event) {
    const previewImg = $('#preview-image');
    const uploadLabel = $('#upload-label');
    const resultImg = $('#result-image');
    

    // Clear previous result image if any
    if (resultImg.length) {
        resultImg.attr('src', '#').addClass('hidden');
    }

    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.attr('src', e.target.result).removeClass('hidden');
            uploadLabel.addClass('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        previewImg.addClass('hidden');
    }
}

// Clear inputs, hide preview and spinner, restore upload label
function resetForm() {
    const form = $('form')[0];
    if (!form) return;

    // Clear file input
    const fileInput = $('#file-upload');
    fileInput.val('');

    // Hide preview and clear src
    const previewImg = $('#preview-image');
    previewImg.addClass('hidden').attr('src', '#');

    // Show upload label
    $('#upload-label').removeClass('hidden');

    // Hide spinner
    $('#spinner').addClass('hidden');

}

// Ensure file input change uses showPreview
$('#file-upload').off('change').on('change', showPreview);

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
