import face_recognition
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def analyze_face(image_path):
    # Load the image
    image = face_recognition.load_image_file(image_path)
    
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    
    if len(face_landmarks_list) == 0:
        return "No face found in the image."
    
    # We'll work with the first face found
    face_landmarks = face_landmarks_list[0]
    
    # Get face encoding
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        return "No face encoding could be generated."
    face_encoding = face_encodings[0]
    
    # Calculate eye distance
    left_eye = np.mean(face_landmarks['left_eye'], axis=0)
    right_eye = np.mean(face_landmarks['right_eye'], axis=0)
    eye_distance = np.linalg.norm(left_eye - right_eye)
    
    # Estimate hair color
    top_of_forehead = face_landmarks['top_lip'][0]  # Using top lip as reference
    hair_sample = image[max(0, int(top_of_forehead[1]) - 50):int(top_of_forehead[1]), 
                        max(0, int(top_of_forehead[0]) - 25):min(image.shape[1], int(top_of_forehead[0]) + 25)]
    
    # Use K-means clustering to find the dominant color
    hair_sample_rgb = Image.fromarray(hair_sample).convert('RGB')
    hair_sample_array = np.array(hair_sample_rgb).reshape(-1, 3)
    if len(hair_sample_array) > 0:
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(hair_sample_array)
        dominant_color = kmeans.cluster_centers_[0]
    else:
        dominant_color = np.array([0, 0, 0])  # Default to black if no sample available
    
    # Very basic hair color classification
    hair_colors = {
        'black': [0, 0, 0],
        'brown': [165, 42, 42],
        'blonde': [255, 215, 0],
        'red': [255, 0, 0],
        'gray': [128, 128, 128]
    }
    hair_color = min(hair_colors, key=lambda x: np.linalg.norm(np.array(hair_colors[x]) - dominant_color))
    
    # Estimate skin tone (using nose tip as reference)
    nose_tip = face_landmarks['nose_tip'][0]
    skin_sample = image[max(0, int(nose_tip[1])-10):min(image.shape[0], int(nose_tip[1])+10), 
                        max(0, int(nose_tip[0])-10):min(image.shape[1], int(nose_tip[0])+10)]
    skin_sample_rgb = Image.fromarray(skin_sample).convert('RGB')
    skin_sample_array = np.array(skin_sample_rgb).reshape(-1, 3)
    if len(skin_sample_array) > 0:
        kmeans.fit(skin_sample_array)
        skin_tone = kmeans.cluster_centers_[0]
    else:
        skin_tone = np.array([0, 0, 0])  # Default if no sample available

    # Estimate age (very rough approximation)
    # This is a placeholder and would need a proper machine learning model for accuracy
    wrinkle_areas = np.concatenate([face_landmarks['left_eye'], face_landmarks['right_eye'], face_landmarks['top_lip']])
    wrinkle_intensity = np.mean(image[wrinkle_areas[:, 1], wrinkle_areas[:, 0], 0])  # Using red channel

    # Check for receding hairline
    forehead_height = face_landmarks['left_eyebrow'][0][1] - top_of_forehead[1]
    receding_hairline = forehead_height > eye_distance * 0.8  # Arbitrary threshold

    # Determine face shape (very basic approximation)
    face_width = face_landmarks['chin'][16][0] - face_landmarks['chin'][0][0]
    face_height = face_landmarks['chin'][8][1] - top_of_forehead[1]
    face_ratio = face_height / face_width
    face_shape = 'oval' if 1.3 <= face_ratio <= 1.7 else 'round' if face_ratio < 1.3 else 'long'

    # Check for beard and mustache
    lower_lip = np.mean(face_landmarks['bottom_lip'], axis=0)
    chin = face_landmarks['chin'][8]
    potential_beard_area = image[int(lower_lip[1]):int(chin[1]), int(chin[0])-20:int(chin[0])+20]
    beard_intensity = np.mean(potential_beard_area)
    has_beard = beard_intensity < 100  # Arbitrary threshold, lower intensity suggests darker area (beard)

    mustache_area = image[int(face_landmarks['nose_tip'][0][1]):int(face_landmarks['top_lip'][0][1]), 
                          int(face_landmarks['nose_tip'][0][0])-20:int(face_landmarks['nose_tip'][0][0])+20]
    mustache_intensity = np.mean(mustache_area)
    has_mustache = mustache_intensity < 100  # Arbitrary threshold

    # Estimate gender (very basic approximation)
    # This is a placeholder and would need a proper machine learning model for accuracy
    jaw_width = face_landmarks['chin'][16][0] - face_landmarks['chin'][0][0]
    brow_to_jaw = face_landmarks['chin'][8][1] - face_landmarks['left_eyebrow'][0][1]
    gender_ratio = jaw_width / brow_to_jaw
    estimated_gender = 'male' if gender_ratio > 1.1 else 'female'  # Very rough approximation

    left_cheek = face_landmarks['left_eyebrow'][-1]
    right_cheek = face_landmarks['right_eyebrow'][-1]
    nose_tip = face_landmarks['nose_tip'][0]
    mouth_left = face_landmarks['top_lip'][0]
    mouth_right = face_landmarks['top_lip'][6]

    cheekbone_width = distance.euclidean(left_cheek, right_cheek)
    mouth_width = distance.euclidean(mouth_left, mouth_right)

    cheekbone_prominence = cheekbone_width / mouth_width

    high_cheekbones = cheekbone_prominence

    # Check for glasses
    eye_region = np.concatenate([face_landmarks['left_eye'], face_landmarks['right_eye']])
    eye_region_mean = np.mean(image[eye_region[:, 1], eye_region[:, 0]])
    has_glasses = eye_region_mean > 200  # Adjust threshold based on image brightness

    # Check for narrow eyes
    left_eye_width = distance.euclidean(face_landmarks['left_eye'][0], face_landmarks['left_eye'][3])
    right_eye_width = distance.euclidean(face_landmarks['right_eye'][0], face_landmarks['right_eye'][3])
    eye_width_ratio = (left_eye_width + right_eye_width) / (2 * eye_distance)
    narrow_eyes = eye_width_ratio

    # Check for baldness
    forehead = image[max(0, int(top_of_forehead[1]) - 50):int(top_of_forehead[1]), 
                     max(0, int(top_of_forehead[0]) - 25):min(image.shape[1], int(top_of_forehead[0]) + 25)]
    forehead_mean = np.mean(forehead)
    bald = forehead_mean > 200 and hair_color in ['gray', 'blonde']
    

    return {
        'eye_distance': float(eye_distance),
        'hair_color': hair_color,
        'skin_tone': skin_tone.tolist(),
        'receding_hairline': receding_hairline,
        'face_shape': face_shape,
        'has_beard': has_beard,
        'has_mustache': has_mustache,
        'estimated_gender': estimated_gender,
        'face_encoding': face_encoding.tolist(),
        'bald': bald,
        'narrow_eyes': narrow_eyes,
        'has_glasses': has_glasses,
        'high_cheekbones': high_cheekbones
    }

def find_closest_match(new_image_path, csv_path):
    df = pd.read_csv(csv_path)
    
    # Analyze the new image
    new_face_data = analyze_face(new_image_path)
    
    if isinstance(new_face_data, str):
        return f"Error analyzing new image: {new_face_data}"
    
    # Extract the face encoding from the new image
    new_face_encoding = np.array(new_face_data['face_encoding'])
    
    # Extract face encodings from the CSV
    existing_face_encodings = df['face_encoding'].apply(eval).tolist()
    
    # Calculate cosine similarity
    similarities = cosine_similarity([new_face_encoding], existing_face_encodings)[0]
    
    # Find the index of the most similar face
    most_similar_index = np.argmax(similarities)
    
    # Get the name and similarity score of the closest match
    closest_match = df.iloc[most_similar_index]['name']
    similarity_score = similarities[most_similar_index]
    
    return {
        'closest_match': closest_match,
        'similarity_score': similarity_score,
        'new_face_data': new_face_data
    }

# Usage
new_image_path = './people/angela_merkel.jpg'
csv_path = 'facial_analysis_results.csv'

result = find_closest_match(new_image_path, csv_path)
print(f"Closest match: {result['closest_match']}")
print(f"Similarity score: {result['similarity_score']}")
print("New face data:")
for key, value in result['new_face_data'].items():
    if key != 'face_encoding':
        print(f"  {key}: {value}")

# print(analyze_face('./people/high_cheekbones.jpg'))