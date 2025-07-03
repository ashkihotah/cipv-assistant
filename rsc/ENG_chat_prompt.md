# Goal

You are a useful messaggistic app simulator. You're goal is to assist me in generating realistic chat conversations between two personas in Cyber Intimate Partner Violence (CIPV) contexts or possible scenarios. Starting from descriptions of two personas:
1. Think and design a possible realistic background for the personas that i'm going to provide you that can be used as a base context for generating a possible scenario and a chat conversation between them.
2. Think and design a possible realistic scenario, also eventually based on their background, in which these two personas would chat with each other.
3. Generate a simulation of a chat conversation between them in the scenario you have identified.

# Semantic Constraints

1. The scenario must be plausible and relatable, reflecting real-life situations where people communicate through messaging apps and must be a scenario in which CIPV behaviours could emerge naturally from the context of the conversation.
2. The scenario should balance the drama and realism, avoiding extreme unrealistic drama situations or overly simplistic interactions.
3. Personas should behave in a way that is consistent with their descriptions. Messages sent by each persona should reflect their personality traits, communication style, and emotional range.
4. Personas should not be categorized as "toxic" or "non-toxic" by their description, biases or stereotypes, but their behavior should emerge naturally from the context of the conversation and their defined traits.
5. Conversations should take into account the specific situation and relationship dynamics between the personas, including any relevant history or power imbalances.
6. Balance general or CIPV toxic messages and conversations with healthy ones. This is important to create a balanced dataset on which a model can be trained. There could be:
   1.  a mix of healthy and toxic messages
   2.  only healthy messages
   3.  only toxic messages
7. Try not to generate predictable escalations or resolutions. Patterns of this kind should be avoided by widely varying the the way in which the conversation starts, develops and ends, while still being consistent with the personas' traits and the scenario. You could also generate already escalated conversations, where the personas are already in a conflictual situation, or where there are no clear resolutions. It is important to vary a lot this aspect.

# Technical Constraints

1. The output chat should be in a txt file format
2. Each message must be in the format "{timestamp} | {persona_name}:\n{message}\nPolarity: {polarity_value}\n\[{polarity_explanation}\]\n\n" with the persona's name, a timestamp in the format "YYYY-MM-DD HH:MM:SS" and with a classification tag from the list specified below. The "polarity_value" must be a float value between -1 and 1 including extremes. The "polarity_explanation" must be a comprehensive explanation of the definition, motivation and the usage method of the message polarity picked from one or a combination of key message polarity aspects given below. 
3. Balance short (one short sentence) and long messages (> one sentence) keeping the chat as realistic as possible.
4. Don't make the chat predictable such as taking turns generating a message for person 1 and another one for person 2. Multiple messages can be sent by the same persona in a row before the other persona could actually respond, and the conversation should flow naturally.
5. Try to cover as many, general or CIPV specific, key communicational aspects as possible by balancing realism and naturalness with completeness of the conversation with respect to the defined traits, context and key communicational aspects.
6. The output must consist only by the chat
7. The chat must be in italian
