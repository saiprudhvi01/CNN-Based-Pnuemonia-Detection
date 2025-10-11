from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, UnidentifiedImageError
import numpy as np
import random
from classify import classify  # Import classify function from classify.py

app = Flask(__name__)

# Pneumonia symptoms and their weights for severity calculation
PNEUMONIA_SYMPTOMS = {
    'fever': {'weight': 0.15, 'threshold': 38.5},
    'cough': {'weight': 0.20, 'threshold': 7},  # days
    'chest_pain': {'weight': 0.18, 'threshold': 5},  # severity 1-10
    'shortness_of_breath': {'weight': 0.25, 'threshold': 7},  # severity 1-10
    'fatigue': {'weight': 0.12, 'threshold': 6},  # severity 1-10
    'sputum': {'weight': 0.10, 'threshold': 4}  # severity 1-10
}

# Risk factors that contribute to pneumonia development
RISK_FACTORS = {
    'smoking': 0.25,
    'age_over_65': 0.20,
    'chronic_disease': 0.18,
    'weakened_immune': 0.15,
    'recent_viral': 0.12,
    'poor_nutrition': 0.10
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/symptom_analysis', methods=['GET', 'POST'])
def symptom_analysis():
    if request.method == 'POST':
        # Get symptom data from form
        symptoms = {}
        for symptom in PNEUMONIA_SYMPTOMS.keys():
            value = request.form.get(f'{symptom}_value', 0)
            try:
                symptoms[symptom] = float(value)
            except ValueError:
                symptoms[symptom] = 0

        # Get risk factors
        risk_factors = {}
        for factor in RISK_FACTORS.keys():
            risk_factors[factor] = 1 if request.form.get(factor) == 'on' else 0

        # Calculate severity and generate analysis
        severity_score, risk_assessment = calculate_pneumonia_severity(symptoms, risk_factors)
        feature_importance = calculate_feature_importance(symptoms, risk_factors)
        explanation = generate_explanation(symptoms, risk_factors, severity_score, feature_importance)

        return render_template('symptom_analysis.html',
                             symptoms=symptoms,
                             risk_factors=risk_factors,
                             severity_score=severity_score,
                             risk_assessment=risk_assessment,
                             feature_importance=feature_importance,
                             explanation=explanation)

    return render_template('symptom_analysis.html')

@app.route('/pneumonia_game', methods=['GET', 'POST'])
def pneumonia_game():
    if request.method == 'POST':
        return handle_game_response()

    # Initialize or get game state
    if request.args.get('start') == 'true':
        return initialize_triage_game()
    elif request.args.get('reset') == 'true':
        return reset_triage_game()

    return get_triage_game_state()

def initialize_triage_game():
    """Initialize the medical triage game session"""
    game_state = {
        'current_question': 0,
        'total_questions': 10,  # Fixed to match actual question count
        'score': 0,
        'responses': {},
        'questions': get_triage_questions(),
        'game_started': True,
        'game_completed': False,
        'assessment_result': None
    }

    return render_template('pneumonia_game.html', game_state=game_state)

def get_triage_game_state():
    """Get current triage game state"""
    return initialize_triage_game()

def reset_triage_game():
    """Reset the triage game"""
    return initialize_triage_game()

def handle_game_response():
    """Handle player's response to triage questions"""
    import json
    from flask import jsonify

    # Get current game state
    game_state = {
        'current_question': int(request.form.get('current_question', 0)),
        'responses': json.loads(request.form.get('responses', '{}')),
        'questions': get_triage_questions()
    }

    # Get the answer for current question
    question_key = game_state['questions'][game_state['current_question']]['key']
    answer = request.form.get('answer', '')

    # Store response
    game_state['responses'][question_key] = answer

    # Debug logging
    print(f"Current question: {game_state['current_question']}")
    print(f"Total questions: {len(game_state['questions'])}")
    print(f"Answer: {answer}")
    print(f"Responses so far: {game_state['responses']}")

    # Move to next question or complete assessment
    if game_state['current_question'] < len(game_state['questions']) - 1:
        game_state['current_question'] += 1
        print(f"Moving to question: {game_state['current_question']}")
        return render_template('pneumonia_game.html', game_state=game_state)
    else:
        # Assessment completed - calculate results
        assessment_result = calculate_pneumonia_assessment(game_state['responses'])
        game_state['assessment_result'] = assessment_result
        game_state['game_completed'] = True

        print(f"Assessment complete. Risk level: {assessment_result['risk_level']}")
        return render_template('pneumonia_game.html', game_state=game_state)

def get_triage_questions():
    """Get the list of simple yes/no triage questions for the game"""
    return [
        {
            'id': 1,
            'key': 'cough',
            'question': 'Do you have a new cough that started recently?',
            'type': 'choice',
            'options': ['Yes', 'No'],
            'points': 2,
            'description': 'New cough is a key pneumonia symptom'
        },
        {
            'id': 2,
            'key': 'fever',
            'question': 'Do you have a fever or feel like you have chills?',
            'type': 'choice',
            'options': ['Yes', 'No'],
            'points': 2,
            'description': 'Fever indicates infection'
        },
        {
            'id': 3,
            'key': 'shortness_of_breath',
            'question': 'Are you feeling short of breath or having trouble breathing?',
            'type': 'choice',
            'options': ['Yes', 'No'],
            'points': 3,
            'description': 'Breathing difficulty suggests lung involvement'
        },
        {
            'id': 4,
            'key': 'chest_pain',
            'question': 'Do you have chest pain when you breathe or cough?',
            'type': 'choice',
            'options': ['Yes', 'No'],
            'points': 1,
            'description': 'Chest pain indicates lung inflammation'
        },
        {
            'id': 5,
            'key': 'productive_sputum',
            'question': 'Do you have colored (green/yellow) or bloody phlegm when you cough?',
            'type': 'choice',
            'options': ['Yes', 'No'],
            'points': 1,
            'description': 'Colored sputum suggests infection'
        },
        {
            'id': 6,
            'key': 'confusion',
            'question': 'Are you feeling confused, disoriented, or unusually sleepy?',
            'type': 'choice',
            'options': ['Yes', 'No'],
            'points': 3,
            'description': 'Confusion requires immediate attention'
        },
        {
            'id': 7,
            'key': 'rigors',
            'question': 'Do you have shaking chills or feel like you\'re shivering uncontrollably?',
            'type': 'choice',
            'options': ['Yes', 'No'],
            'points': 1,
            'description': 'Shaking chills indicate severe infection'
        },
        {
            'id': 8,
            'key': 'age_risk',
            'question': 'Are you 65 years old or older?',
            'type': 'choice',
            'options': ['Yes', 'No'],
            'points': 2,
            'description': 'Age 65+ increases pneumonia risk'
        },
        {
            'id': 9,
            'key': 'underlying_conditions',
            'question': 'Do you have asthma, COPD, heart disease, diabetes, or a weakened immune system?',
            'type': 'choice',
            'options': ['Yes', 'No'],
            'points': 2,
            'description': 'Underlying conditions increase risk'
        },
        {
            'id': 10,
            'key': 'smoking',
            'question': 'Do you currently smoke or have you smoked in the past?',
            'type': 'choice',
            'options': ['Yes', 'No'],
            'points': 1,
            'description': 'Smoking damages lungs and increases risk'
        }
    ]

def calculate_pneumonia_assessment(responses):
    """Calculate pneumonia risk assessment based on simple yes/no responses"""
    score = 0
    risk_factors = []
    red_flags = []

    # Simple yes/no scoring
    if responses.get('cough') == 'Yes':
        score += 2
        risk_factors.append(f"New cough (+{2} points) - Key pneumonia symptom")

    if responses.get('fever') == 'Yes':
        score += 2
        risk_factors.append(f"Fever/chills (+{2} points) - Indicates infection")

    if responses.get('shortness_of_breath') == 'Yes':
        score += 3
        risk_factors.append(f"Shortness of breath (+{3} points) - Lung involvement")

    if responses.get('chest_pain') == 'Yes':
        score += 1
        risk_factors.append(f"Chest pain (+{1} point) - Lung inflammation")

    if responses.get('productive_sputum') == 'Yes':
        score += 1
        risk_factors.append(f"Colored sputum (+{1} point) - Infection indicator")

    if responses.get('confusion') == 'Yes':
        score += 3
        risk_factors.append(f"Confusion (+{3} points) - Requires immediate attention")
        red_flags.append("Confusion requires immediate medical attention")

    if responses.get('rigors') == 'Yes':
        score += 1
        risk_factors.append(f"Shaking chills (+{1} point) - Severe infection indicator")

    if responses.get('age_risk') == 'Yes':
        score += 2
        risk_factors.append(f"Age 65+ (+{2} points) - Higher risk group")

    if responses.get('underlying_conditions') == 'Yes':
        score += 2
        risk_factors.append(f"Underlying conditions (+{2} points) - Compromised immunity")

    if responses.get('smoking') == 'Yes':
        score += 1
        risk_factors.append(f"Smoking history (+{1} point) - Lung damage")

    # Determine risk level
    if red_flags or score >= 8:
        risk_level = "HIGH"
        recommendation = "URGENT: Seek immediate medical attention. Call emergency services if you have severe shortness of breath, confusion, or chest pain."
        urgency = "emergency"
    elif score >= 4:
        risk_level = "MODERATE"
        recommendation = "Contact healthcare provider for evaluation within 24-48 hours. Consider chest X-ray and clinical examination."
        urgency = "outpatient"
    else:
        risk_level = "LOW"
        recommendation = "Unlikely to be pneumonia. Monitor symptoms for 48 hours. Seek care if symptoms worsen or new symptoms develop."
        urgency = "home"

    return {
        'score': score,
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'red_flags': red_flags,
        'recommendation': recommendation,
        'urgency': urgency,
        'disclaimer': "This is a screening tool only, not a medical diagnosis. Always consult healthcare professionals for proper evaluation."
    }

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return "Bad Request: Please select a file to upload.", 400

        file = request.files['file']

        try:
            img = Image.open(file.stream).convert('RGB')
            img_array = np.array(img)
            prediction = classify(img_array)
        except UnidentifiedImageError:
            return "Bad Request: Invalid image file. Please upload a valid image.", 400
        except Exception as e:
            # Log the specific error for debugging
            print(f"Prediction error: {str(e)}")
            return f"Error during prediction: {str(e)}", 500

        return render_template('index.html', prediction=prediction)
    
    return render_template('index.html')

def calculate_pneumonia_severity(symptoms, risk_factors):
    """Calculate pneumonia severity based on symptoms and risk factors"""
    severity_score = 0
    risk_score = 0

    # Calculate symptom-based severity
    for symptom, value in symptoms.items():
        if symptom in PNEUMONIA_SYMPTOMS:
            weight = PNEUMONIA_SYMPTOMS[symptom]['weight']
            threshold = PNEUMONIA_SYMPTOMS[symptom]['threshold']

            if symptom == 'fever' and value >= threshold:
                severity_score += weight * min(value / 40, 1)  # Normalize fever
            elif symptom != 'fever' and value >= threshold:
                severity_score += weight * (value / 10)  # Normalize other symptoms

    # Calculate risk factor contribution
    for factor, present in risk_factors.items():
        if present and factor in RISK_FACTORS:
            risk_score += RISK_FACTORS[factor]

    # Combine scores
    total_score = min(severity_score + (risk_score * 0.3), 1.0)

    # Determine risk level
    if total_score < 0.3:
        risk_level = "Low"
    elif total_score < 0.6:
        risk_level = "Moderate"
    elif total_score < 0.8:
        risk_level = "High"
    else:
        risk_level = "Very High"

    return total_score, risk_level

def calculate_feature_importance(symptoms, risk_factors):
    """Calculate which features are most important for the current case"""
    importance = {}

    # Calculate symptom importance
    for symptom, value in symptoms.items():
        if symptom in PNEUMONIA_SYMPTOMS:
            weight = PNEUMONIA_SYMPTOMS[symptom]['weight']
            threshold = PNEUMONIA_SYMPTOMS[symptom]['threshold']

            if symptom == 'fever':
                importance_score = weight * min(value / 40, 1) if value >= threshold else 0
            else:
                importance_score = weight * (value / 10) if value >= threshold else 0

            importance[symptom] = importance_score

    # Add risk factor importance
    for factor, present in risk_factors.items():
        if present and factor in RISK_FACTORS:
            importance[factor] = RISK_FACTORS[factor] * 0.3

    return importance

def generate_explanation(symptoms, risk_factors, severity_score, feature_importance):
    """Generate dynamic explanation based on the most important features"""
    # Get top 3 most important features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:3]

    explanations = {
        'fever': "High fever indicates a strong immune response to infection, often one of the first signs of pneumonia.",
        'cough': "Persistent coughing helps clear mucus and irritants from the airways but can also spread infection.",
        'chest_pain': "Chest pain during breathing suggests inflammation of the lung lining, a classic pneumonia symptom.",
        'shortness_of_breath': "Difficulty breathing indicates the lungs are not functioning properly due to infection and inflammation.",
        'fatigue': "Extreme tiredness results from the body diverting energy to fight the infection rather than normal activities.",
        'sputum': "Coughing up phlegm or mucus is the body's attempt to clear infected material from the lungs.",
        'smoking': "Smoking damages lung tissue and weakens the immune system, making pneumonia more likely and severe.",
        'age_over_65': "Older adults have weaker immune systems and are more susceptible to respiratory infections.",
        'chronic_disease': "Existing health conditions compromise the immune system and lung function.",
        'weakened_immune': "A weakened immune system cannot effectively fight off respiratory infections.",
        'recent_viral': "Recent viral infections can damage lung tissue, creating an environment for bacterial pneumonia.",
        'poor_nutrition': "Poor nutrition weakens the immune system and reduces the body's ability to fight infections."
    }

    # Generate dynamic explanation based on top features
    primary_feature = top_features[0][0] if top_features else 'general_symptoms'

    if severity_score < 0.3:
        severity_text = "mild respiratory infection"
        urgency = "generally not urgent, but monitor symptoms closely"
        treatment = "rest, stay hydrated, and use over-the-counter medications for symptom relief"
    elif severity_score < 0.6:
        severity_text = "moderate pneumonia risk"
        urgency = "should be evaluated by a healthcare provider"
        treatment = "may require antibiotics and closer monitoring"
    elif severity_score < 0.8:
        severity_text = "significant pneumonia concern"
        urgency = "requires prompt medical attention"
        treatment = "likely needs prescription medication and possibly hospitalization"
    else:
        severity_text = "severe pneumonia risk"
        urgency = "requires immediate medical attention"
        treatment = "may require hospitalization and intensive treatment"

    # Create personalized explanation
    explanation_parts = [
        f"Based on your symptoms, you appear to have a {severity_text}. ",

        f"The most significant contributing factor appears to be {primary_feature.replace('_', ' ')}, which {explanations.get(primary_feature, 'is contributing significantly to your symptoms')}. ",

        f"This condition {urgency}. ",

        f"Treatment typically involves {treatment}. ",

        "To prevent pneumonia: maintain good hand hygiene, avoid smoking, get vaccinated against flu and pneumonia, maintain a healthy diet, and manage chronic conditions properly. ",

        "If left untreated, pneumonia can lead to serious complications including respiratory failure, sepsis, or lung abscess. Early intervention significantly improves outcomes."
    ]

    return ''.join(explanation_parts)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5100)
