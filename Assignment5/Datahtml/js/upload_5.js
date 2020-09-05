function uploadAndClassifyImage(){
	var fileInput = document.getElementById('HumanPoseUpload').files;
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
		url: 'https://wv8f5xwqk4.execute-api.ap-south-1.amazonaws.com/dev/estimate',
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})	
	.done(function(response){
		console.log(response);
		var image = document.getElementById('output9');
		image.src = "https://session5-body-pose.s3.amazonaws.com/body-pose.jpg?t=" + new Date().getTime();
	})
	.fail(function(){alert("There was an error while sending prediction request to get human pose");});
	};
	
    $('#btnHumanPoseUpload').click(uploadAndClassifyImage);