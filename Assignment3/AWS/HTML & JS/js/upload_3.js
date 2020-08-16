function uploadAndClassifyImage(){
	var fileInput = document.getElementById('FaceMergeFileUpload').files;
	if(!fileInput.length){
		return alert('please choose o upload first image');
	}
	var fileInput1 = document.getElementById('FaceMergeFileUpload1').files;
	if(!fileInput1.length){
		return alert('please choose to upload second image');
	}
	
	var file = fileInput[0];
	var filename = file.name;
	var file1 = fileInput1[0];
	var filename1 = file1.name;

	var formData = new FormData();
	formData.append(filename, file);
	formData.append(filename1, file1);

	console.log(filename);

	$.ajax({
		async: true,
		crossDomain: true,
		method: 'POST',
		url: 'https://uygiqqxfqc.execute-api.ap-south-1.amazonaws.com/dev/align',
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})	
	.done(function(response){
		console.log(response);
		var image = document.getElementById('output6');
		image.src = "https://session3-face-alignment-face-swap.s3.amazonaws.com/face-swap.png?t=" + new Date().getTime();
	})
	.fail(function(){alert("There was an error while sending prediction request to align model");});
	};
	
    $('#btnFaceMergeUpload').click(uploadAndClassifyImage);