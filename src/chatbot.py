"""
Simple Chatbot for Traffic Prediction MLOps Application
======================================================

Provides explanations of processes in simple, non-technical language.
Uses rule-based responses and templates for reliability.
"""

import logging
import streamlit as st
from typing import Dict, List, Any
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficPredictionChatbot:
    """
    A simple chatbot that explains traffic prediction processes in plain language.
    Uses rule-based responses for reliability without external AI dependencies.
    """
    
    def __init__(self):
        self.conversation_history = []
        self.current_step = None
        self.explanations = self._load_explanations()
        self.keywords = self._load_keywords()
        
    def _load_explanations(self) -> Dict[str, str]:
        """Load pre-written explanations for each process step."""
        return {
            'file_upload': """
            **Ce qui se passe maintenant : Téléchargement du fichier**
            
            Vous téléchargez votre fichier de données de trafic. C'est comme donner au système un tableau 
            avec des informations sur les flux de circulation - par exemple la vitesse des voitures, 
            combien de voitures sont passées, et à quel moment cela s'est produit.
            
            Le système vérifie si votre fichier contient les bonnes informations nécessaires pour faire des prédictions.
            """,
            
            'data_validation': """
            **Ce qui se passe maintenant : Vérification des données**
            
            Le système examine vos données pour s'assurer qu'elles sont complètes et cohérentes. 
            C'est comme un inspecteur qualité qui vérifie que :
            - Toutes les informations importantes sont présentes
            - Les chiffres semblent raisonnables (pas de voitures à 800 km/h !)
            - Les données sont bien organisées
            
            Cela permet d'assurer des prédictions fiables par la suite.
            """,
            
            'coordinate_detection': """
            **Ce qui se passe maintenant : Recherche des informations de localisation**
            
            Le système recherche des informations de localisation dans vos données - comme des coordonnées GPS ou des adresses.
            Cela permet de savoir exactement où les mesures de trafic ont été prises.
            
            C'est comme placer des épingles sur une carte pour montrer l'origine de chaque relevé de trafic.
            """,
            
            'data_preprocessing': """
            **Ce qui se passe maintenant : Préparation des données**
            
            C'est comme nettoyer et organiser vos données pour que l'ordinateur puisse mieux les comprendre.
            Le système :
            - Corrige les informations manquantes ou incorrectes
            - Convertit tout dans un format standard
            - Crée de nouvelles informations utiles à partir de ce que vous avez fourni
            
            C'est similaire à ranger un bureau avant de commencer un travail important.
            """,
            
            'feature_engineering': """
            **Ce qui se passe maintenant : Création de caractéristiques intelligentes**
            
            Le système crée de nouvelles informations à partir de vos données originales pour améliorer les prédictions. Par exemple :
            - Si vous avez la vitesse et l'heure, il peut détecter les heures de pointe
            - Il peut regrouper des conditions de trafic similaires
            - Il crée des "empreintes" de trafic pour identifier des schémas
            
            C'est comme un détective qui trouve des indices cachés.
            """,
            
            'model_training': """
            **Ce qui se passe maintenant : Apprentissage du système**
            
            Le système apprend à partir de vos données de trafic pour faire des prédictions. C'est comme former un élève en lui montrant de nombreux exemples :
            - "Quand le trafic ressemble à ÇA, cela conduit généralement à ÇA"
            - Il essaie différentes méthodes pour trouver la meilleure façon de prédire
            - Il se teste pour voir la précision de ses prédictions
            
            Ce processus permet au système de mieux anticiper les futurs schémas de circulation.
            """,
            
            'model_evaluation': """
            **Ce qui se passe maintenant : Test des performances**
            
            Le système vérifie ce qu'il a appris en testant ses prédictions sur des données qu'il n'a jamais vues. C'est comme passer un examen final :
            - À quelle fréquence devine-t-il correctement ?
            - Quelle est la marge d'erreur lorsqu'il se trompe ?
            - Quelle méthode fonctionne le mieux pour vos données ?
            
            Cela nous aide à savoir si nous pouvons faire confiance aux prédictions.
            """,
            
            'visualization': """
            **Ce qui se passe maintenant : Création de rapports visuels**
            
            Le système crée des cartes et des graphiques pour vous montrer les résultats de façon claire :
            - Cartes indiquant où les problèmes de trafic sont probables
            - Graphiques montrant l'évolution du trafic dans le temps
            - Tableaux comparant différentes méthodes de prédiction
            
            Les rapports visuels rendent l'information complexe plus facile à comprendre et à utiliser.
            """,
            
            'prediction': """
            **Ce qui se passe maintenant : Prédictions**
            
            Le système entraîné réalise maintenant des prédictions sur les conditions de trafic à venir. 
            Il utilise tout ce qu'il a appris à partir de vos données pour anticiper :
            - Où des embouteillages pourraient se produire
            - Quand les routes seront les plus encombrées
            - Quelles routes pourraient être de meilleures alternatives
            
            Ces prédictions peuvent vous aider à planifier et à prendre des décisions.
            """
        }
    
    def _load_keywords(self) -> Dict[str, List[str]]:
        """Load keywords that trigger specific explanations."""
        return {
            'accuracy': ['accuracy', 'correct', 'right', 'precise', 'error'],
            'data_quality': ['quality', 'clean', 'missing', 'bad data', 'incomplete'],
            'machine_learning': ['ai', 'machine learning', 'algorithm', 'model', 'training'],
            'predictions': ['predict', 'forecast', 'future', 'estimate'],
            'visualization': ['map', 'chart', 'graph', 'visual', 'see'],
            'location': ['gps', 'coordinates', 'location', 'where', 'place'],
            'traffic': ['traffic', 'cars', 'vehicles', 'congestion', 'jam'],
            'time': ['time', 'when', 'hour', 'day', 'schedule', 'pattern']
        }
    
    def explain_step(self, step_name: str) -> str:
        """Get explanation for a specific step."""
        self.current_step = step_name
        explanation = self.explanations.get(step_name, "I'm working on processing your traffic data to make better predictions.")
        
        # Add to conversation history
        self.conversation_history.append({
            'type': 'explanation',
            'step': step_name,
            'content': explanation
        })
        
        return explanation
    
    def chat(self, user_message: str) -> str:
        """Process user message and provide appropriate response."""
        user_message = user_message.lower().strip()
        
        # Add user message to history
        self.conversation_history.append({
            'type': 'user_question',
            'content': user_message
        })
        
        # Find appropriate response
        response = self._generate_response(user_message)
        
        # Add response to history
        self.conversation_history.append({
            'type': 'bot_response',
            'content': response
        })
        
        return response
    
    def _generate_response(self, message: str) -> str:
        """Generate response based on user message."""
        
        # Check for specific questions about current step
        if self.current_step and any(word in message for word in ['what', 'why', 'how', 'explain', 'quoi', 'pourquoi', 'comment', 'explique']):
            if 'happening' in message or 'doing' in message or 'passe' in message or 'fait' in message:
                return self.explanations.get(self.current_step, "Je traite vos données de trafic.")
        
        # Keyword-based responses
        if any(word in message for word in self.keywords['accuracy']):
            return """
            **À propos de la précision :**
            
            La précision indique à quelle fréquence nos prédictions sont correctes. C'est comme une prévision météo :
            - Une grande précision (90%+) signifie que nous avons raison la plupart du temps
            - Une précision plus faible signifie que les prédictions sont moins fiables
            - Nous testons la précision avec des données que le système n'a jamais vues
            
            Pour la prédiction du trafic, une bonne précision vous aide à faire confiance aux prévisions pour planifier vos trajets.
            """
        
        elif any(word in message for word in self.keywords['data_quality']):
            return """
            **À propos de la qualité des données :**
            
            Une bonne qualité des données, c'est comme avoir des informations claires et complètes :
            - Complètes : Pas de détails importants manquants
            - Précises : Les chiffres sont cohérents (pas de valeurs impossibles)
            - Cohérentes : Tout suit le même format
            
            Une mauvaise qualité des données donne des prédictions peu fiables, c'est pourquoi nous vérifions et nettoyons tout d'abord.
            """
        
        elif any(word in message for word in self.keywords['machine_learning']):
            return """
            **À propos de l'apprentissage automatique :**
            
            L'apprentissage automatique, c'est comme apprendre à un ordinateur à reconnaître des schémas :
            - On lui montre beaucoup d'exemples de données de trafic
            - Il apprend quelles conditions mènent à quels résultats
            - Ensuite, il peut faire des suppositions éclairées sur de nouvelles situations
            
            C'est comme vous, qui apprenez à anticiper le trafic après avoir pris la même route plusieurs fois.
            """
        
        elif any(word in message for word in self.keywords['predictions']):
            return """
            **À propos des prédictions :**
            
            Nos prédictions sont des estimations sur le trafic futur basées sur les schémas trouvés :
            - Elles sont plus précises pour des conditions similaires à vos données d'entraînement
            - La météo, les événements ou des circonstances inhabituelles peuvent influencer la précision
            - Utilisez-les comme des guides utiles, pas comme des certitudes absolues
            
            Pensez-y comme des prévisions météo : très utiles pour planifier, mais il faut toujours rester flexible.
            """
        
        elif any(word in message for word in self.keywords['visualization']):
            return """
            **À propos des visualisations :**
            
            Nous créons des cartes et des graphiques pour rendre les données complexes faciles à comprendre :
            - Les cartes montrent OÙ les problèmes de trafic sont probables
            - Les graphiques montrent QUAND les schémas de trafic changent
            - Les tableaux aident à comparer différentes méthodes de prédiction
            
            L'information visuelle est souvent plus facile à comprendre que des tableaux de chiffres.
            """
        
        # General help responses
        elif any(word in message for word in ['help', 'what can you do', 'explain', 'aide', 'peux-tu', 'explique']):
            return """
            **Je suis là pour vous aider à comprendre le processus de prédiction du trafic !**
            
            Je peux expliquer :
            - Ce qui se passe à chaque étape
            - Pourquoi certaines étapes sont importantes
            - Ce que signifient les résultats
            - Comment interpréter les visualisations
            
            Posez-moi des questions comme :
            - "Que se passe-t-il maintenant ?"
            - "Pourquoi cette étape est-elle importante ?"
            - "Que signifient ces résultats ?"
            - "Quelle est la précision des prédictions ?"
            """
        
        # Default response
        else:
            return """
            Je suis là pour vous expliquer simplement le processus de prédiction du trafic.
            
            Vous pouvez me demander :
            - Ce qui se passe actuellement
            - Pourquoi certaines étapes sont importantes
            - Ce que signifient les résultats
            - Comment utiliser les prédictions
            
            Que souhaitez-vous savoir ?
            """
    
    def get_progress_summary(self, completed_steps: List[str]) -> str:
        """Provide a summary of completed steps."""
        if not completed_steps:
            return "We're just getting started with your traffic data analysis!"
        
        summary = "**Here's what we've accomplished so far:**\n\n"
        
        step_descriptions = {
            'file_upload': '✅ Received your traffic data file',
            'data_validation': '✅ Checked data quality and completeness',
            'coordinate_detection': '✅ Found location information in your data',
            'data_preprocessing': '✅ Cleaned and organized the data',
            'feature_engineering': '✅ Created smart features for better predictions',
            'model_training': '✅ Trained the prediction system',
            'model_evaluation': '✅ Tested prediction accuracy',
            'visualization': '✅ Created maps and charts',
            'prediction': '✅ Generated traffic forecasts'
        }
        
        for step in completed_steps:
            if step in step_descriptions:
                summary += f"{step_descriptions[step]}\n"
        
        summary += f"\n**Total steps completed: {len(completed_steps)}**"
        return summary
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.current_step = None


def create_chatbot_interface():
    """Create the Streamlit chatbot interface."""
    
    # Initialize chatbot in session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = TrafficPredictionChatbot()
    
    st.markdown("### 🤖 Assistant IA")
    st.markdown("*Posez-moi vos questions sur le processus de prédiction du trafic !*")
    
    # Chat history display
    if st.session_state.chatbot.conversation_history:
        st.markdown("**Conversation :**")
        
        # Show recent conversation (last 6 messages)
        recent_history = st.session_state.chatbot.conversation_history[-6:]
        
        for entry in recent_history:
            if entry['type'] == 'user_question':
                st.markdown(f"**Vous :** {entry['content']}")
            elif entry['type'] == 'bot_response':
                st.markdown(f"**Assistant :** {entry['content']}")
            elif entry['type'] == 'explanation':
                st.info(entry['content'])
    
    # User input
    user_input = st.text_input(
        "Posez-moi une question sur le processus :",
        placeholder="ex : Que se passe-t-il maintenant ? Pourquoi cette étape est-elle importante ?",
        key="chatbot_input"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Demander", key="ask_button"):
            if user_input:
                response = st.session_state.chatbot.chat(user_input)
                st.rerun()
    
    with col2:
        if st.button("Effacer la discussion", key="clear_chat"):
            st.session_state.chatbot.clear_history()
            st.rerun()
    
    # Quick help buttons
    st.markdown("**Questions rapides :**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Que se passe-t-il ?", key="whats_happening"):
            response = st.session_state.chatbot.chat("Que se passe-t-il maintenant ?")
            st.rerun()
    
    with col2:
        if st.button("Quelle précision ?", key="accuracy_question"):
            response = st.session_state.chatbot.chat("Quelle est la précision des prédictions ?")
            st.rerun()
    
    with col3:
        if st.button("Aide", key="help_button"):
            response = st.session_state.chatbot.chat("aide")
            st.rerun()


def notify_step_completion(step_name: str, details: str = None):
    """Notify the chatbot when a step is completed."""
    if 'chatbot' in st.session_state:
        explanation = st.session_state.chatbot.explain_step(step_name)
        
        # Show notification
        with st.expander("🤖 AI Assistant Explanation", expanded=True):
            st.markdown(explanation)
            if details:
                st.markdown(f"**Additional details:** {details}")