let selectedMessages = new Set();
let messages = [];
let messageIdCounter = 0;
let alignClass = {};

// Close modal when clicking outside of it
window.onclick = function(event) {
    const modal = document.getElementById('errorModal');
    if (event.target === modal) {
        closeErrorModal();
    }
};

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    // Sync JavaScript state with persisted HTML input values
    syncPersonNamesFromInputs();

    renderChat();
    
    // Add enter key handler for message input
    document.getElementById('messageInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});

function syncPersonNamesFromInputs() {
    const nameInputs = document.querySelectorAll('.person-name-input');

    // Build the alignClass Dictionary
    nameInputs.forEach((input, index) => {
        const inputValue = input.value.trim();
        if (inputValue) {
            alignClass[inputValue] = `person-${String.fromCharCode(97 + index)}`;
            input.setAttribute('data-old-value', inputValue); // Store the old value for future reference
        }
    });
}

function showErrorModal(message) {
    const modal = document.getElementById('errorModal');
    const messageElement = document.getElementById('errorMessage');
    messageElement.textContent = message;
    modal.style.display = 'block';
    
    // Prevent body scrolling when modal is open
    document.body.style.overflow = 'hidden';
}

function closeErrorModal() {
    const modal = document.getElementById('errorModal');
    modal.style.display = 'none';
    
    // Restore body scrolling
    document.body.style.overflow = 'auto';
}

function updatePersonValue(inputElement) {
    const newValue = inputElement.value.trim();
    const oldValue = inputElement.getAttribute('data-old-value');

    // check if the new value is empty
    if (!newValue) {
        showErrorModal("Person name cannot be empty.");
        inputElement.value = oldValue;  // Revert to old value
        return;
    }

    // Check if the new value already exists
    if (alignClass[newValue] && newValue !== oldValue) {
        // If it exists, alert the user and revert the change
        showErrorModal(`Person name "${newValue}" is already in use. Please choose a different name.`);
        inputElement.value = oldValue;  // Revert to old value
        return;
    }

    // Update the radio button value
    // radioButtons[radioIndex].value = newValue;

    // update person align class mapping
    if (alignClass[oldValue]) {
        alignClass[newValue] = alignClass[oldValue];
        delete alignClass[oldValue];
    }

    // Update existing messages with the new person name
    messages.forEach(msg => {
        if (msg.person === oldValue) {
            msg.person = newValue;
        }
    });

    // Update the stored old value for next time
    inputElement.setAttribute('data-old-value', newValue);

    // Re-render chat to reflect changes
    renderChat();
}

function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    const person_input = document.querySelector('input[name="person"]:checked');
    const person_input_id = person_input.value == "person-A-name-input" ? 0 : 1;
    const person = document.querySelector(`input[id="${person_input.value}"]`).value;

    if (!message) return;

    // Create new message
    const timestamp = new Date().toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    
    const newMessage = {
        'id': messageIdCounter++,
        'person': person,
        'person_id': person_input_id,
        'message': message,
        'timestamp': timestamp
    };

    // Add message to array
    messages.push(newMessage);

    // Clear input and re-render
    messageInput.value = '';
    renderChat(true);
}

function toggleMessageSelection(messageId) {
    if (selectedMessages.has(messageId)) {
        selectedMessages.delete(messageId);
    } else {
        selectedMessages.add(messageId);
    }
    renderChat();
}

function selectAll() {
    selectedMessages.clear();
    messages.forEach(msg => selectedMessages.add(msg.id));
    renderChat();
}

function clearSelection() {
    selectedMessages.clear();
    renderChat();
}

function getSelectedMessages() {
    return messages.filter(msg => selectedMessages.has(msg.id));
}

function addPolaritiesToMessages(selectedMsgs, polarities) {
    // Add polarity data to the corresponding messages
    selectedMsgs.forEach((msg, index) => {
        msg.polarity = polarities[index];
    });
    
    // Re-render chat to show the polarity labels
    renderChat();
}

function getPolarityColor(polarity) {
    // Clamp polarity to [-1, 1] range
    polarity = Math.max(-1, Math.min(1, polarity));
    
    let r, g, b;
    
    if (polarity < 0) {
        // Red to Yellow (-1 to 0)
        const ratio = (polarity + 1); // Convert from [-1,0] to [0,1]
        r = 255;
        g = Math.round(255 * ratio);
        b = 0;
    } else {
        // Yellow to Green (0 to 1)
        const ratio = polarity; // Already in [0,1] range
        r = Math.round(255 * (1 - ratio));
        g = 255;
        b = 0;
    }
    
    return `rgb(${r}, ${g}, ${b})`;
}

async function runAnalysis(type) {
    const resultsContent = document.getElementById('resultsContent');
    const selectedMsgs = getSelectedMessages();
    // const task = document.querySelector('input[name="task"]:checked').value;
    const task = 'toxicity'; // Hardcoded for demo
    const labels = document.querySelector('input[name="labels"]:checked').value;
    const preprocessor = document.querySelector('input[name="preprocessor"]:checked').value;
    const model = document.querySelector(`input[name="${type}-model"]:checked`).value;
    
    // Get target message position
    let targetMessagePosition = 0; // Default to first message
    if (selectedMsgs.length > 1) {
        const targetRadio = document.querySelector('input[name="targetMessage"]:checked');
        if (targetRadio) {
            targetMessagePosition = parseInt(targetRadio.value);
        }
    }

    try {
        const response = await fetch(`/api/analysis/${type}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                'task': task,
                'labels': labels,
                'preprocessor': preprocessor,
                'model': model,
                'messages': selectedMsgs,
                'target_idx': targetMessagePosition
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (type === 'message-regression') {
            // Parse the polarities and add them to messages
            const polarities = result.output; // Assuming this is an array of polarity values
            if (polarities.length === selectedMsgs.length) {
                addPolaritiesToMessages(selectedMsgs, polarities);
            } else {
                selectedMsgs[targetMessagePosition].polarity = polarities[0];
                renderChat();
            }
            resultsContent.textContent = `Added polarity labels to ${selectedMsgs.length} messages`;
        } else if (type === 'chat-explanation') {
            resultsContent.textContent = result.output;
        } else if (type === 'messages-regression-explanation') {
            resultsContent.textContent = result.output;
            const polarities = result.polarities; // Assuming this is an array of polarity values
            addPolaritiesToMessages(selectedMsgs, polarities);
        } else {
            resultsContent.innerHTML = convertMarkdownToHtml(`**Predicted Label:** ${result.label}\n**Probabilities:** ${result.probabilities}`);
        }
    } catch (error) {
        console.error('Error running analysis:', error);
        resultsContent.innerHTML = '<em style="color: red;">Error running analysis: ' + error.message + '</em>';
    }
}

function convertMarkdownToHtml(text) {
    return text
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^\*\*(.+)\*\*$/gm, '<h3>$1</h3>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
        .replace(/^(\d+\. .+)$/gm, '<li>$1</li>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
}

function renderChat(scrollToBottom = false) {
    const chatContainer = document.getElementById('chatContainer');
    
    if (messages.length === 0) {
        chatContainer.innerHTML = '<div class="empty-chat">No messages yet. Start a conversation!</div>';
        return;
    }

    const selectedMsgs = getSelectedMessages();
    // const showTargetRadios = selectedMsgs.length > 1;

    let chatHtml = '';
    messages.forEach(msg => {
        const isSelected = selectedMessages.has(msg.id);
        const selectedIndex = selectedMsgs.findIndex(selectedMsg => selectedMsg.id === msg.id);

        // Create polarity label if polarity exists
        let polarityLabel = '';
        if (msg.polarity !== undefined) {
            const polarityColor = getPolarityColor(msg.polarity);
            polarityLabel = `
                <span class="polarity-label" style="background-color: ${polarityColor};">
                    Polarity: ${msg.polarity.toFixed(2)}
                </span>
            `;
        }

        // Create target radio button if this message is selected and there are multiple selections
        let targetRadio = '';
        if (isSelected) { //  && showTargetRadios
            const isFirstSelected = selectedIndex === 0;
            targetRadio = `
                <div class="target-radio-container">
                    <input type="radio" 
                           id="target-${msg.id}" 
                           name="targetMessage" 
                           value="${selectedIndex}" 
                           ${isFirstSelected ? 'checked' : ''}
                           onclick="event.stopPropagation();">
                    <label for="target-${msg.id}" onclick="event.stopPropagation();">Target</label>
                </div>
            `;
        }

        chatHtml += `
            <div class="message-div ${alignClass[msg.person]}">
                <div class="message-bubble ${isSelected ? 'selected' : ''}"
                        onclick="toggleMessageSelection(${msg.id})">
                    <div class="message-person">${msg.person}</div>
                    <div class="message-content">${msg.message}</div>
                    <div class="message-timestamp">${targetRadio}${polarityLabel}${msg.timestamp}</div>
                </div>
            </div>
        `;
    });

    chatContainer.innerHTML = chatHtml;
    // Only scroll to bottom when explicitly requested
    if (scrollToBottom) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}