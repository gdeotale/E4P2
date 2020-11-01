function uploadAndClassifyImage(){
	var fileInput = document.getElementById('SentenceUpload').files;
	if(!fileInput.length){
		return alert('please write a sentence to upload');
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
		url: 'https://9nnncm80a9.execute-api.ap-south-1.amazonaws.com/dev/classify',
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})	
	.done(function(response){
		console.log(response);
		document.getElementById('result9').textContent = response;
	})
	.fail(function(){alert("There was an error while sending prediction request to Textnet model");});
	};
	
    $('#btnSentenceUpload').click(uploadAndClassifyImage);