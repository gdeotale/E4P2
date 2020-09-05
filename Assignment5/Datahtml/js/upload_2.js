function uploadAndClassifyImage(){
	var fileInput = document.getElementById('FaceFileUpload').files;
	if(!fileInput.length){
		return alert('please choose file to upload first');
	}
	
	var file = fileInput[0];
	var filename = file.name;

	var formData = new FormData();
	formData.append(filename, file);

	console.log(filename);

	$.ajax({
		async: true,
		crossDomain: true,
		method: 'POST',
		url: 'https://d7p4azpba6.execute-api.ap-south-1.amazonaws.com/dev/align',
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})	
	.done(function(response){
		console.log(response);
		var image = document.getElementById('output3');
		image.src = "https://session3--face-alignment-face-swap.s3.amazonaws.com/test.png?t=" + new Date().getTime();
	})
	.fail(function(){alert("There was an error while sending prediction request to align model");});
	};
	
    $('#btnFaceFileUpload').click(uploadAndClassifyImage);