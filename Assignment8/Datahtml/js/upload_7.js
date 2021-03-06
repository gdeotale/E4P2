function uploadAndClassifyImage(){
	var fileInput = document.getElementById('ReconstructCarUpload').files;
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
		url: 'https://mibr0zsht3.execute-api.ap-south-1.amazonaws.com/dev/getcars',
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})	
	.done(function(response){
		console.log(response);
		var image = document.getElementById('output_7_');
		image.src = "https://gdeotale-session7-cars.s3.amazonaws.com/cars.jpg?t=" + new Date().getTime();
	})
	.fail(function(){alert("There was an error while sending prediction request to align model");});
	};
	
    $('#btnReconstructCarUpload').click(uploadAndClassifyImage);