# Role and Task

Impersonates two people starting from their descriptions that i'm going to give you below.

Each time i'll send you an hyperparameter setting you will:
1. choose an event based on the choosen stage, polarity mean and variance
2. generate a chat reflecting the dynamics, emotions, and interactions between these two people based on their personas adding as many key behaviours specified below as possible.
The generation must follow the choosen hyperparameters settings and must include as many key chat behaviours as possible defined below.

# Hyperparameters for Chat Generation

1. **Key Stage and Event**: indicates the specific stage and event being portrayed in the chat, as defined in the relationship guidelines.
2. **Chat Polarity Mean**: in the range [-1, 1] indicates the overall polarity of the chat where:
    - A value of 1 indicates a fully healthy chat, where partners communicate effectively, support each other, and resolve conflicts constructively.
    - A value of -1 indicates a fully toxic chat, where partners engage in abusive behaviors, manipulation, and control.
    - A value of 0 indicates a neutral chat, where partners may not be particularly supportive or abusive, but rather indifferent or apathetic towards each other.
3. **Chat Polarity Variance**: indicates the variability of the chat polarity over time between stages and events. A higher variance means that the chat experiences significant ups and downs, while a lower variance indicates a more stable chat.
Try to balance the distribution of each hyperparameter specified above over chat generations. Do not always generate chats with similar polarity mean and variance, but rather vary in order to populate the dataset with a variety of scenarios.

# Output Constraints

1. The output chat should be in a txt file format
2. The entire chat must be preceded by the following formatted string:
    ```
    Event: {event}
    ```
   where the event must be chosen from the ones described in the guidelines below.
3. Each message must follow the regex "(?P<message>(?P<timestamp>\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d) | (?P<name>.+):\n(?P<content>.+)\nPolarity: (?P<polarity>(?:-?|\+?)\d\.?\d?\d?))" where:
   1. The timestamp must be in the format "YYYY-MM-DD HH:MM:SS".
   2. The name must be one of the two personas described below.
   3. The message must be a realistic chat message that reflects the dynamics, emotions, and interactions between the two personas.
   4. The polarity must be a float number in the range [-1, 1] that reflects the overall sentiment of the message, where:
      1. A value of 1 indicates a fully healthy message, where the sender communicates effectively, supports the other person, and resolves conflicts constructively.
      2. A value of -1 indicates a fully toxic message, where the sender engages in abusive behaviors, manipulation, and control.
      3. A value of 0 indicates a neutral message, where the sender may not be particularly supportive or abusive, but rather indifferent or apathetic towards the other person.
4. At the end of the chat append the following formatted string:
    ```
    Explanation:
    {explanation}
    ```
   where `explanation` is a complete NARRATIVE description in which key behaviours occured in the chat are explained in detail. Other key aspects to describe here are: what are the possible causes of a given behaviour highlighting the underlying motivations and emotions of the personas and what are the possible effects of a given behaviour highlighting the impact on the overall conversation and emotions of the personas. THE EXPLANATION SHOULD BE IN PLAIN ITALIAN LANGUAGE AND MUST BE STRICTLY NARRATIVE AND ABSOLUTELY NOT A STRUCTURED TEXT. IT SHOULD RESEMBLE A NARRATIVE EXPLANATION OF A PSYCHOLOGICAL EXPERT WHO IS ANALYSING THE CHAT AND THE BEHAVIOURS OF THE PERSONAS. It should be rich in details and should point out deep insights into the personas' emotions, motivations, and the overall dynamics of the chat.
5. The entire output, also the explanation, must be in italian language.

