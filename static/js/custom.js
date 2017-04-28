$(document).ready(function() {

	$('#newsletter-form').submit(function() {
		var buttonCopy = $('#newsletter-form button').html(),
		errorMessage = $('#newsletter-form button').data('error-message'),
		sendingMessage = $('#newsletter-form button').data('sending-message'),
		okMessage = $('#newsletter-form button').data('ok-message'),
		hasError = false;
		$('#newsletter-form .error-message').remove();
		$('#newsletter-form .requiredField').each(function() {
			$(this).removeClass('inputError');
			if($.trim($(this).val()) == '') {
				var errorText = $(this).data('error-empty');
				$(this).parents('.myform').prepend('<span class="error-message" style="display:none;">'+errorText+'.</span>').find('.error-message').fadeIn('fast');
				$(this).addClass('inputError');
				hasError = true;
			} else if($(this).is("input[type='email']") || $(this).attr('name')==='email') {
				var emailReg = /^([\w-\.]+@([\w-]+\.)+[\w-]{2,4})?$/;
				if(!emailReg.test($.trim($(this).val()))) {
					var invalidEmail = $(this).data('error-invalid');
					$(this).parents('.myform').prepend('<span class="error-message" style="display:none;">'+invalidEmail+'.</span>').find('.error-message').fadeIn('fast');
					$(this).addClass('inputError');
					hasError = true;
				}
			}			
		});
		
		if(hasError) {
			$('#newsletter-form button').html(errorMessage).addClass('btn-error');
			
			setTimeout(function(){
				$('#newsletter-form button').removeClass('btn-error').html(buttonCopy);
				
			},2000);
		}else {
			$('#newsletter-form button').html(sendingMessage);
			
			var formInput = $(this).serialize();
			var link = $(this).attr('action');	
			$.ajax({
				type: 'POST',	
				url: link,
				data: formInput
			}).done(function(data) {
			 // $( this ).addClass( "done" );
			  $('#newsletter-form button').html(okMessage);
			  setTimeout(function(){
					$('#newsletter-form button').html(buttonCopy);

				},2000);
			})
			.fail(function() {
				$('#newsletter-form button').html('Save error');
    			setTimeout(function(){
					$('#newsletter-form button').html(buttonCopy);
				},2000);
  			});

		}
		return false;	
	});



	var config = {
    countdown: {
        year: 2015,
        month: 8,
        day: 24,
        hour: 10,
        minute: 55,
        second: 12
    }};
	
    var date = new Date(config.countdown.year,
                        config.countdown.month - 1,
                        config.countdown.day,
                        config.countdown.hour,
                        config.countdown.minute,
                        config.countdown.second),
        $countdownNumbers = {
            days: $('#countdown-days'),
            hours: $('#countdown-hours'),
            minutes: $('#countdown-minutes'),
            seconds: $('#countdown-seconds')
        };

    $('#countdown').countdown(date).on('update.countdown', function(event) {
        $countdownNumbers.days.text(event.offset.totalDays);
        $countdownNumbers.hours.text(('0' + event.offset.hours).slice(-2));
        $countdownNumbers.minutes.text(('0' + event.offset.minutes).slice(-2));
        $countdownNumbers.seconds.text(('0' + event.offset.seconds).slice(-2));
    });

});