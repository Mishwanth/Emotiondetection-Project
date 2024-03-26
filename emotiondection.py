import cv2
from deepface import DeepFace
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import matplotlib.pyplot as plt
from pydub import AudioSegment
import pygame.mixer



smtp_server = 'smtp.gmail.com'
smtp_port = 587
sender_email = 'sainath1733749@gmail.com'
email_password = 'fltcupjkoneslxaw'

server = smtplib.SMTP(smtp_server, smtp_port)
server.starttls()
server.login(sender_email, email_password)



#function to send an email to parent/user 
def send_email(emotion):
    subject = 'Most Expressed Emotion'
    body = f'The most expressed emotion currently is: {emotion}'
    receiver_email = 'leomishwa@gmail.com'

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject

    text = MIMEText(body, 'plain')
    message.attach(text)

    server.sendmail(sender_email, receiver_email, message.as_string())



pygame.mixer.init()
pygame.mixer.music.load(r"C:\Users\MISHWANTH\Music\twinkle.mp3")
playing = False


#function to call the send_mail function and play music when threshold is reached
def mostexpressed():
    playing = False
    treshold = 300
    valuelist = emotions.values()
    if emotions['sad'] > treshold and emotions['sad'] == max(valuelist):
        #print("Sent")
        pygame.mixer.music.play()
        playing = True
        send_email('sad')
    elif emotions['fear'] > treshold and emotions['fear'] == max(valuelist):
        #print("Sent")
        pygame.mixer.music.play()
        playing = True
        send_email('fear')
    else:
        if playing:
            pygame.mixer.music.stop()
            playing = False





emotions = {
    'happy': 0,
    'sad': 0,
    'angry': 0,
    'surprise': 0,
    'fear': 0,
    'neutral': 0,
    'disgust': 0
}



frame_data = []

desired_fps = 30
delay_time = int(1000 / desired_fps)
webCam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cycle = 0



while True:
    cycle = cycle + 1
    success, frame = webCam.read()

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    emotion = ''


    if len(faces) > 0:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        k = result[0]
        emotion = k['dominant_emotion']
        emotions[emotion] += 1

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)



    else:
        cv2.putText(frame, "Face is Covered or Not in Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    cv2.imshow('Webcam Output', frame)

    del frame

    frame_info = {
        'timestamp': time.time(),
        'emotion': emotion,
        'face_count': len(faces),
    }
    frame_data.append(frame_info)



    if cycle == 600:
        mostexpressed()

        # Create a histogram of emotions
        emotion_labels, emotion_counts = zip(*emotions.items())

        plt.bar(emotion_labels, emotion_counts)
        plt.xlabel('Emotion')
        plt.ylabel('Frequency')
        plt.title('Emotion Distribution')
        plt.show()

        cycle = 0
        continue



    key = cv2.waitKey(delay_time)
    if key == ord('q'):
        break

webCam.release()
cv2.destroyAllWindows()
