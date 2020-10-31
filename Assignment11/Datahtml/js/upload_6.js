function uploadAndClassifyImage(){

	$.ajax({
		async: true,
		crossDomain: true,
		method: 'POST',
		url: 'https://at058si8uc.execute-api.ap-south-1.amazonaws.com/dev/getcars',
		data: '', 
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})	
	.done(function(response){
		console.log(response);
		var image = document.getElementById('output11');
		image.src = "https://gdeotale-session6-cars.s3.amazonaws.com/cars.jpg?t=" + new Date().getTime();
	})
	.fail(function(){alert("There was an error predicting cars");});
	};
	
    $('#btnGetCarUpload').click(uploadAndClassifyImage);