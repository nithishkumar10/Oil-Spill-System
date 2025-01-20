from django.shortcuts import render

# Create your views here.


from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from userapp.models import *
import urllib.request
import pandas as pd
import time
from adminapp.models import *
import urllib.parse
import random
import ssl


# Create your views here.






# ------------------------------------------------------------------------------------------------



#userviews



import pytz
from datetime import datetime



from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .models import PredictionResult

def user_dashboard(req):
    # Fetch predictions, feedback count, and user data
    prediction_count = PredictionResult.objects.all().count()
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id=user_id)
    Feedbacks_users_count = Feedback.objects.all().count()
    all_users_count = UserModel.objects.all().count()

    # Fetch all prediction results
    prediction_results = PredictionResult.objects.all()

    return render(
        req, "user/User-Dashboard.html",
        {
            "predictions": prediction_count,
            "user_name": user.user_name,
            "feedback_count": Feedbacks_users_count,
            "all_users_count": all_users_count,
            "prediction_results": prediction_results
        },
    )

# View to delete a prediction
def delete_prediction(request, prediction_id):
    # Get the prediction object by ID
    prediction = get_object_or_404(PredictionResult, id=prediction_id)

    # Delete the prediction object
    prediction.delete()

    # Redirect back to the dashboard or wherever necessary
    return redirect('user_dashboard')  # Redirect to the user dashboard or a success page







def user_profile(req):
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id=user_id)
    if req.method == "POST":
        user_name = req.POST.get("username")
        user_age = req.POST.get("age")
        user_phone = req.POST.get("mobile_number")
        user_email = req.POST.get("email")
        user_password = req.POST.get("Password")
        user_address = req.POST.get("address")

        user.user_name = user_name
        user.user_age = user_age
        user.user_address = user_address
        user.user_contact = user_phone
        user.user_email = user_email
        user.user_password = user_password

        if len(req.FILES) != 0:
            image = req.FILES["profilepic"]
            user.user_image = image
            user.save()
            messages.success(req, "Updated Successfully.")
        else:
            user.save()
            messages.success(req, "Updated Successfully.")

    context = {"i": user}
    return render(req, "user/User-Profile2.html", context)


# ---------------------------------------------------------------------------------------------------------------


import os
from django.core.files.storage import default_storage
from django.contrib import messages
from django.conf import settings
from django.contrib import messages


# Create your views here.
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from userapp.models import *
import urllib.request
import pandas as pd
import time

from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.files.storage import default_storage
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.applications.densenet import preprocess_input
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
import os
import random  # For generating a 4-digit pin
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from mainapp.models import *
# Load the model and other necessary imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input

# Assuming you already have a loaded model, e.g., `model`
model = load_model('Oil Spil Dataset/oil_densenet.h5')

def predictionss(image_path):
    """This function makes predictions on the image at the provided path."""
    img = image.load_img(image_path, target_size=(224, 224))  # Load image and resize
    i = image.img_to_array(img)  # Convert image to array
    i = np.expand_dims(i, axis=0)  # Expand dimensions to match model input
    img = preprocess_input(i)  # Preprocess image (e.g., normalization)
    pred = np.argmax(model.predict(img), axis=1)  # Get the predicted class
    return pred[0]  # Return the predicted class index (ensure it’s an int)
import json  # Add this import
from django.http import JsonResponse
from django.core.files.base import ContentFile
from django.core.mail import send_mail
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import datetime
import os
from django.conf import settings
from .models import PredictionResult
import base64
# Load the pre-trained model
model = load_model('Oil Spil Dataset/oil_densenet.h5')

def predictionss(image_data):
    """This function makes predictions on the image provided."""
    # Decode the image from base64
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    img = img.resize((224, 224))
    
    # Convert to array and preprocess
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Normalize if required
    
    # Predict the class
    pred = np.argmax(model.predict(img_array), axis=1)
    return pred[0]

def live_camera_prediction(request):
    """Handle the live cam image predictions."""
    if request.method == "POST":
        # Receive the image data from the webcam
        data = json.loads(request.body)  # Ensure json is imported
        image_data = data.get('image')
        
        # Make prediction
        predicted_class = predictionss(image_data)

        # If oil spill detected, send email alert
        if predicted_class == 1:
            email = data.get('email')
            if email and '@' in email:
                mail_message = "Oil Spill Detected!"
                send_mail(
                    "Alert!",
                    mail_message,
                    settings.EMAIL_HOST_USER,
                    [email],
                )

        return JsonResponse({"prediction": predicted_class}, status=200)

    return JsonResponse({"error": "Invalid request"}, status=400)

import os
import datetime
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.contrib import messages
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from .models import PredictionResult  # Assuming you created this model

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponse
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
from django.conf import settings

import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import datetime
from django.conf import settings

def generate_pdf(predicted_class, email, uploaded_image_url):
    # Define the path where the PDF should be saved
    pdf_directory = os.path.join(settings.MEDIA_ROOT, 'prediction_pdfs')
    
    # Ensure the directory exists
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)  # Create the directory if it doesn't exist
    
    # Generate a unique filename for the PDF
    file_path = os.path.join(pdf_directory, f"prediction_result_{str(datetime.datetime.now().timestamp())}.pdf")
    
    # Create the PDF
    pdf = canvas.Canvas(file_path, pagesize=letter)
    pdf.setFont("Helvetica", 14)

    # Set up a margin for layout
    margin_left = 72
    margin_top = 750
    
    # Title/Header - Centered and Bold
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString((letter[0] - pdf.stringWidth("Prediction Result for Oil Spill Detection", "Helvetica-Bold", 16)) / 2, margin_top, "Prediction Result for Oil Spill Detection")
    
    # Move down and add Prediction
    margin_top -= 30
    prediction_text = "Prediction: " + ("Oil Spill" if predicted_class == 1 else "No Oil Spill")
    pdf.drawString((letter[0] - pdf.stringWidth(prediction_text, "Helvetica-Bold", 16)) / 2, margin_top, prediction_text)
    
    # Move down for email and date/time (Italicized)
    margin_top -= 40
    pdf.setFont("Helvetica-Oblique", 12)  # Italicize the text
    pdf.drawString(margin_left, margin_top, f"Email: {email}")
    margin_top -= 20
    pdf.drawString(margin_left, margin_top, f"Date & Time: {str(datetime.datetime.now())}")

    # Reset font for the remaining content
    pdf.setFont("Helvetica", 12)
    margin_top -= 40  # Move down for the next section
    
    # Image URL
    pdf.drawString(margin_left, margin_top, f"Image URL: {uploaded_image_url}")
    
    # Move down and start adding precautions content
    margin_top -= 60
    precautions = """
    Oil spills can cause significant harm to marine ecosystems. 
    If an oil spill is detected, immediate action should be taken:
    - Stop the source of the spill if safe to do so.
    - Contain the spill using barriers and absorbent materials.
    - Contact relevant authorities for further assistance.
    """
    
    # Add precaution text with line wrapping
    text_object = pdf.beginText(margin_left, margin_top)
    text_object.setFont("Helvetica", 12)
    text_object.setTextOrigin(margin_left, margin_top)
    text_object.setLeading(14)  # Space between lines
    text_object.textLines(precautions)  # Automatically wraps the text
    
    # Draw the precaution text block
    pdf.drawText(text_object)
    
    # Add footer with city and current details
    margin_top -= 120  # Move down for footer
    pdf.setFont("Helvetica", 10)
    pdf.drawString(margin_left, margin_top, f"City: Hyderabad")
    margin_top -= 20
    pdf.drawString(margin_left, margin_top, f"Date & Time of Detection: {str(datetime.datetime.now())}")

    # Save the PDF to the specified path
    pdf.save()

    # Return the relative file path for storing in the database
    return os.path.relpath(file_path, settings.MEDIA_ROOT)


def Classification(request):
    result = {"message": "No image uploaded"}
    uploaded_image_url = None

    if request.method == "POST" and 'image' in request.FILES:
        uploaded_image = request.FILES['image']
        
        # Save the uploaded image file
        file_path = default_storage.save(uploaded_image.name, uploaded_image)
        path = os.path.join(settings.MEDIA_ROOT, file_path)
        uploaded_image_url = default_storage.url(file_path)

        # Step 1: Make a prediction for the uploaded image
        predicted_class = predictionss(path)

        # Convert predicted_class to a regular int before saving it in the session
        predicted_class = int(predicted_class)

        # Step 2: If the predicted class is 1 (Oil Spill), send an email
        if predicted_class == 1:
            email = request.POST.get("email")  # Replace with actual user email logic
            if email and '@' in email:
                number = 'Alert!'  # Generate a 4-digit pin
                mail_message = f"Oil Spill Has Been Detected {number}"
                send_mail(
                    "Alert!",  # Subject of the email
                    mail_message,     # Body of the email
                    settings.EMAIL_HOST_USER,  # Sender's email
                    [email],  # Recipient email
                )
                request.session["user_email"] = email  # Store the user's email in the session
            else:
                messages.error(request, "Invalid email address provided.")
        
        # Step 3: Generate and save the PDF
        pdf_file_path = generate_pdf(predicted_class, email, uploaded_image_url)

        # Step 4: Save prediction details into the database
        prediction = PredictionResult(
            predicted_class=predicted_class,
            email=email,
            image_url=uploaded_image_url,
            pdf_file=pdf_file_path,  # Store the relative path to the PDF
            timestamp=datetime.datetime.now()
        )
        prediction.save()

        # Store the result and image URL in the session
        request.session['result'] = {"predicted_class": predicted_class}
        request.session['uploaded_image_url'] = uploaded_image_url  # Store the image URL

        # Add a success message with the prediction result
        messages.warning(request, f'Alert! Predicted class: {predicted_class}. Oil Spill Detection')

        return redirect('Classification_result')  # Redirect to the result view
    
    return render(request, 'user/Prediction.html', {'result': result, 'uploaded_image_url': uploaded_image_url})

from django.contrib import messages

def Classification_result(request):
    result = request.session.get('result', {"message": "No result available"})
    uploaded_image_url = request.session.get('uploaded_image_url', None)
    
    # Retrieve the saved prediction record
    prediction = PredictionResult.objects.last()  # Or use the specific query to get this prediction
    pdf_url = None
    if prediction:
        pdf_url = default_storage.url(prediction.pdf_file)  # Get the URL for downloading the PDF

    # Check if the result is 1 and add a success message
    if result and result.get('prediction') == 1:  # Replace 'prediction' with the correct key if needed
        messages.warning(request, "Prediction result is Oil Spill - Alert !")

    return render(request, 'user/Prediction-result.html', {
        'result': result,
        'uploaded_image_url': uploaded_image_url,
        'pdf_url': pdf_url  # Send PDF URL to be displayed for download
    })




# ------------------------------------------------------------------------------


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


from django.contrib import messages
from django.shortcuts import redirect, render
from .models import Feedback, UserModel  # Make sure to import your models


def user_feedback(req):
    id = req.session["user_id"]
    user = UserModel.objects.get(user_id=id)
    
    if req.method == "POST":
        rating = req.POST.get("rating")
        review = req.POST.get("review")
        
        # Sentiment analysis
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(review)
        
        if score["compound"] > 0 and score["compound"] <= 0.5:
            sentiment = "positive"
        elif score["compound"] >= 0.5:
            sentiment = "very positive"
        elif score["compound"] < -0.5:
            sentiment = "negative"
        elif score["compound"] < 0 and score["compound"] >= -0.5:
            sentiment = "very negative"
        else:
            sentiment = "neutral"
        
        # Create the feedback
        Feedback.objects.create(
            Rating=rating,
            Review=review,
            Sentiment=sentiment,
            Reviewer=user
        )
        
        messages.success(req, "Feedback recorded")
        return redirect("user_feedback")  # Redirecting to the same page
    
    return render(req, "user/User-Feedback.html")



from django.utils import timezone


def user_logout(req):
    if "user_id" in req.session:
        view_id = req.session["user_id"]
        try:
            user = UserModel.objects.get(user_id=view_id)
            user.Last_Login_Time = timezone.now().time()
            user.Last_Login_Date = timezone.now().date()
            user.save()
            messages.info(req, "You are logged out.")
        except UserModel.DoesNotExist:
            pass
    req.session.flush()
    return redirect("index")



def user_login(req):
    if req.method == "POST":
        user_email = req.POST.get("email")
        user_password = req.POST.get("password")
        print(user_email, user_password)

        try:
            users_data = UserModel.objects.filter(user_email=user_email)
            if not users_data.exists():
                messages.error(req, "User does not exist")
                return redirect("user_login")

            for user_data in users_data:
                if user_data.user_password == user_password:
                    if (
                        user_data.Otp_Status == "verified"
                        and user_data.User_Status == "accepted"
                    ):
                        req.session["user_id"] = user_data.user_id
                        messages.success(req, "You are logged in..")
                        user_data.No_Of_Times_Login += 1
                        user_data.save()
                        return redirect("user_dashboard")
                    elif (
                        user_data.Otp_Status == "verified"
                        and user_data.User_Status == "pending"
                    ):
                        messages.info(req, "Your Status is in pending")
                        return redirect("user_login")
                    else:
                        messages.warning(req, "verifyOTP...!")
                        req.session["user_email"] = user_data.user_email
                        return redirect("otp")
                else:
                    messages.error(req, "Incorrect credentials...!")
                    return redirect("user_login")

            # Handle the case where no user data matched the password
            messages.error(req, "Incorrect credentials...!")
            return redirect("user_login")
        except Exception as e:
            print(e)
            messages.error(req, "An error occurred. Please try again later.")
            return redirect("user_login")

    return render(req, "user/user-login.html")

import pickle
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages



from django.contrib import messages



