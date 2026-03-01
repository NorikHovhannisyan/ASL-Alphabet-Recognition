import pickle
from keras.models import load_model
import numpy as np
import cv2

model = load_model('asl_model.h5')

with open('label.pkl', 'rb') as file:
    label = pickle.load(file)
    
cap = cv2.VideoCapture(0)
print(f"Enter 'q' for quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (50, 50), (350, 350), (255, 0, 0), 2)
    roi = frame[100:350, 100:350]
    
    # Նախամշակում (ճիշտ այնպես, ինչպես արել էիր Kaggle-ում)
    img = cv2.resize(roi, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 64, 64, 1) # Ավելացնում ենք batch չափողականությունը

    # Կանխատեսում
    prediction = model.predict(img, verbose=0)
    label_index = np.argmax(prediction)
    
    # Օգտագործում ենք քո բեռնած 'label' (LabelEncoder-ը) տառը ստանալու համար
    # Եթե label-ը սովորական Python dict է կամ list, օգտագործիր label[label_index]
    # Եթե այն LabelEncoder օբյեկտ է, ապա՝ label.inverse_transform([label_index])[0]
    try:
        predicted_char = label.inverse_transform([label_index])[0]
    except:
        predicted_char = label[label_index]

    # Արդյունքը տպում ենք էկրանին
    cv2.putText(frame, f"Letter: {predicted_char}", (100, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Ցուցադրում ենք պատուհանը
    cv2.imshow('What AI Sees', cv2.resize(img[0].reshape(64, 64), (200, 200)))
    cv2.imshow('ASL Detector', frame)

    # 'q' սեղմելիս փակվում է
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
