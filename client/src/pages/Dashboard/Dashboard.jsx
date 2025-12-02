import React, { useState, useEffect, useRef } from 'react';
import { useSelector } from 'react-redux';
import './Dashboard.css'; // Import the CSS

import axios from 'axios';

// --- Constants ---
const API_BASE_URL = import.meta.env.VITE_PYTHON_API_URL;
const DB_NAME = 'EchoesChatHistory_v3';
const DB_VERSION = 1;

console.log("this is the API_BASE_URL", API_BASE_URL)

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

// Add request interceptor to include X-User-ID
apiClient.interceptors.request.use((config) => {
  const userId = localStorage.getItem('echoes_user_id');
  if (userId) {
    config.headers['X-User-ID'] = userId;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

function Dashboard() {

  const { username } = useSelector((state) => state.auth);

  // --- State ---
  // The 'userId' state is now driven by the 'username' from Redux
  const [userId, setUserId] = useState(localStorage.getItem('echoes_user_id') || '');

  const [characters, setCharacters] = useState([]);
  const [activeCharacter, setActiveCharacter] = useState(null);
  const [messages, setMessages] = useState([]);

  // Removed modal states, as auth is handled by Redux
  // const [isUserIdModalOpen, setUserIdModalOpen] = useState(true); 
  const [isCreateModalOpen, setCreateModalOpen] = useState(false);
  const [isPaletteOpen, setPaletteOpen] = useState(false);
  const [isProfilePanelOpen, setProfilePanelOpen] = useState(false);

  const [isThinking, setThinking] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [formFeedback, setFormFeedback] = useState('');

  const [connectionConnected, setConnectionConnected] = useState(false);
  const [connectionTitle, setConnectionTitle] = useState("Backend Connection Status");

  const [auroraClass, setAuroraClass] = useState('');
  const [messageInput, setMessageInput] = useState('');
  const [commandInput, setCommandInput] = useState('');

  const [createFormState, setCreateFormState] = useState({
    name: '',
    description: '',
    my_name: ''
  });
  const [createFile, setCreateFile] = useState(null);

  // --- Refs ---
  const dbRef = useRef(null);
  const chatMessagesRef = useRef(null);
  const commandInputRef = useRef(null);

  // --- API Functions (using Axios) ---

  // fetchCharacters is now an async function defined within the component
  const fetchCharacters = async (currentUserId) => {
    if (!currentUserId) return; // Don't fetch if no user
    try {
      // Use the apiClient instance. The 'X-User-ID' header is added by the interceptor.
      const response = await apiClient.get('/characters');

      console.log("this is the response", response)

      // Ensure response.data is an array
      setCharacters(Array.isArray(response.data) ? response.data : []);
      setConnectionConnected(true);
      setConnectionTitle("Backend Connected");
    } catch (error) {
      console.error('Failed to fetch characters:', error);
      // alert("Connection Error: Could not reach backend.");
      setConnectionConnected(false);
      setConnectionTitle(`Backend Disconnected: ${error.message}`);
    }
  };

  // --- IndexedDB Functions ---
  const initDB = () => {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains('conversations')) {
          const store = db.createObjectStore('conversations', { keyPath: 'id', autoIncrement: true });
          store.createIndex('character_key', ['userId', 'charName'], { unique: false });
        }
      };

      request.onsuccess = (event) => {
        dbRef.current = event.target.result;
        resolve(event.target.result);
      };

      request.onerror = (event) => {
        console.error("IndexedDB error:", event.target.errorCode);
        reject(event.target.errorCode);
      };
    });
  };

  const addMessageToDB = async (messageData) => {
    if (!dbRef.current || !activeCharacter) return;
    const transaction = dbRef.current.transaction(['conversations'], 'readwrite');
    const store = transaction.objectStore('conversations');
    store.add({
      userId: userId, // This uses the component's 'userId' state
      charName: activeCharacter.name,
      ...messageData
    });
  };

  const loadHistoryFromDB = (charName, currentUserId) => {
    return new Promise((resolve) => {
      if (!dbRef.current) return resolve([]);
      const transaction = dbRef.current.transaction(['conversations'], 'readonly');
      const store = transaction.objectStore('conversations');
      const index = store.index('character_key');
      const keyRange = IDBKeyRange.only([currentUserId, charName]);

      const request = index.getAll(keyRange);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve([]);
    });
  };

  const clearChatHistory = async (characterName, currentUserId) => {
    if (!dbRef.current) return;
    const transaction = dbRef.current.transaction(['conversations'], 'readwrite');
    const store = transaction.objectStore('conversations');
    const index = store.index('character_key');
    const keyRange = IDBKeyRange.only([currentUserId, characterName]);

    const request = index.openCursor(keyRange);
    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor) {
        store.delete(cursor.primaryKey);
        cursor.continue();
      } else {
        if (activeCharacter && activeCharacter.name === characterName) {
          setMessages([{
            text: `Chat history with ${characterName} has been cleared.`,
            sender: 'bot',
            avatarInitial: characterName.charAt(0)
          }]);
        }
      }
    };
  };

  // --- Event Handlers ---

  const handleSetActiveCharacter = async (charName) => {
    const character = characters.find(c => c.name === charName);
    if (character) {
      setActiveCharacter(character);
      const history = await loadHistoryFromDB(charName, userId); // Uses 'userId' state

      if (history.length > 0) {
        setMessages(history);
      } else {
        setMessages([{
          text: `You are now chatting with ${character.name}. ${character.description}`,
          sender: 'bot',
          avatarInitial: character.name.charAt(0)
        }]);
      }

      setPaletteOpen(false);
      setProfilePanelOpen(false); // Close profile panel when switching
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    const text = messageInput.trim();
    if (!text || !activeCharacter) return;

    const userMessage = { text, sender: 'user', avatarInitial: 'U', timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMessage]);
    addMessageToDB(userMessage);
    setMessageInput('');
    setThinking(true);

    try {
      // Refactored to Axios
      const response = await apiClient.post('/chat', {
        character_name: activeCharacter.name,
        message: text
      });
      // The interceptor adds the 'X-User-ID' header

      const chatResponse = response.data; // Data is directly on response.data
      const botMessage = {
        text: chatResponse.content,
        sender: 'bot',
        avatarInitial: activeCharacter.name.charAt(0),
        emotion: chatResponse.emotion,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, botMessage]);
      addMessageToDB(botMessage);

      if (chatResponse.emotion) {
        setAuroraClass(chatResponse.emotion.toLowerCase());
        setCharacters(prev => prev.map(c =>
          c.name === activeCharacter.name
            ? { ...c, emotion: chatResponse.emotion }
            : c
        ));
      }

    } catch (error) {
      const errorMessageText = error.response?.data?.detail || error.message || 'API response error';
      const errorMessage = {
        text: `Error: ${errorMessageText}`,
        sender: 'bot',
        avatarInitial: '!',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setThinking(false);
    }
  };

  const handleCreateFormChange = (e) => {
    const { name, value } = e.target;
    setCreateFormState(prev => ({ ...prev, [name]: value }));
  };

  const handleFileChange = (e) => {
    setCreateFile(e.target.files[0]);
  };

  const handleCreateCharacter = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('name', createFormState.name);
    formData.append('description', createFormState.description);
    formData.append('my_name', createFormState.my_name);
    if (createFile) {
      formData.append('file', createFile);
    } else {
      setFormFeedback("Error: Chat history file is required.");
      return;
    }

    setIsCreating(true);
    setFormFeedback('');

    try {
      // Refactored to Axios
      // The interceptor adds the 'X-User-ID' header
      // Axios automatically sets 'Content-Type' to 'multipart/form-data'
      const response = await apiClient.post('/characters', formData);

      const newChar = response.data; // Data is directly on response.data
      alert('Character created successfully!');
      await fetchCharacters(userId); // Re-fetch characters
      closeCreateModal();
      handleSetActiveCharacter(newChar.name);
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to create character';
      setFormFeedback(`Error: ${errorMessage}`);
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteCharacter = async (charName) => {
    if (window.confirm(`Are you sure you want to permanently delete ${charName}? This action cannot be undone.`)) {
      try {
        // Refactored to Axios
        // The interceptor adds the 'X-User-ID' header
        const response = await apiClient.delete(`/characters/${charName}`);

        if (response.status !== 204) { // Check for 204 No Content
          throw new Error("Failed to delete on server.");
        }

        setCharacters(prev => prev.filter(c => c.name !== charName));
        clearChatHistory(charName, userId);

        if (activeCharacter && activeCharacter.name === charName) {
          setActiveCharacter(null);
        }
        setProfilePanelOpen(false);
        alert(`${charName} has been deleted.`);
      } catch (error) {
        const errorMessage = error.response?.data?.detail || error.message || 'Error deleting character';
        console.error('Error deleting character:', error);
        alert(`Error: ${errorMessage}`);
      }
    }
  };

  const handleClearHistory = (charName) => {
    if (window.confirm(`Are you sure you want to clear your chat history with ${charName}?`)) {
      clearChatHistory(charName, userId);
    }
  };

  const closeCreateModal = () => {
    setCreateModalOpen(false);
    setFormFeedback('');
    setCreateFormState({ name: '', description: '', my_name: '' });
    setCreateFile(null);
  };

  const openCreateModal = () => {
    setPaletteOpen(false);
    setTimeout(() => setCreateModalOpen(true), 300); // Wait for palette to close
  };

  // --- Effects ---

  // Effect to sync Redux username to local state and localStorage
  useEffect(() => {
    if (username) {
      console.log('[App.js] Auth state received:', username);
      setUserId(username);
      localStorage.setItem('echoes_user_id', username);
    }
  }, [username]);

  // Effect to fetch characters when userId is set (either from auth or localStorage)
  useEffect(() => {
    if (userId) {
      fetchCharacters(userId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]); // Only run when userId changes

  // Initial load (DB)
  useEffect(() => {
    initDB();
  }, []);

  // Global keydown listeners
  useEffect(() => {
    const handleKeydown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setPaletteOpen(true);
      }
      if (e.key === 'Escape') {
        setPaletteOpen(false);
        setCreateModalOpen(false);
        setProfilePanelOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages]);

  // Focus command input
  useEffect(() => {
    if (isPaletteOpen && commandInputRef.current) {
      commandInputRef.current.focus();
    }
  }, [isPaletteOpen]);

  // --- Render Logic ---
  const commands = [
    { name: 'Create New Character', action: 'create_char', icon: '✨', handler: openCreateModal },
    ...characters.map(c => ({
      name: `Chat with ${c.name}`,
      action: 'chat',
      icon: '💬',
      charName: c.name,
      handler: () => handleSetActiveCharacter(c.name)
    }))
  ];

  const filteredCommands = commandInput
    ? commands.filter(item => item.name.toLowerCase().includes(commandInput.toLowerCase()))
    : commands;


  return (
    <>
      <div className={`aurora-background ${auroraClass}`} id="aurora-background"></div>

      <div className="app-container">
        {/* Sidebar */}
        <aside className="sidebar" id="sidebar">
          <header className="sidebar-header">
            <h1>Digital Me</h1>
            <div
              className={`connection-status ${connectionConnected ? 'connected' : ''}`}
              id="connection-status"
              title={connectionTitle}
            ></div>
          </header>
          <div className="sidebar-content" id="sidebar-content">
            <h3>Characters</h3>
            {characters.length === 0 ? (
              <div className="empty-state-action">
                <p>No characters yet.</p><br />
                <button id="empty-state-create-btn" onClick={() => setCreateModalOpen(true)}>Create First Character</button>
              </div>
            ) : (
              characters.map(char => (
                <div
                  key={char.name}
                  className={`character-list-item ${activeCharacter?.name === char.name ? 'active' : ''}`}
                  onClick={() => handleSetActiveCharacter(char.name)}
                >
                  <div className={`avatar ${char.emotion ? char.emotion.toLowerCase() : ''}`}>
                    {char.name.charAt(0)}
                  </div>
                  <span>{char.name}</span>
                </div>
              ))
            )}
          </div>
          <div className="sidebar-footer" id="sidebar-footer">
            {userId && (
              <div id="user-id-display" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', wordBreak: 'break-all', marginTop: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                User ID: {userId}
              </div>
            )}
          </div>
        </aside>

        {/* Main Canvas */}
        <main className="main-canvas">
          {!activeCharacter ? (
            <div id="welcome-view" className="view active">
              <h2>Welcome</h2>
              <p>{userId ? "Select a character to start chatting." : "Please load a User ID to begin."}</p>
            </div>
          ) : (
            <div id="chat-view" className="view active">
              <header className="chat-header" id="chat-header" onClick={() => setProfilePanelOpen(true)}>
                <div className="avatar" id="chat-header-avatar">{activeCharacter.name.charAt(0)}</div>
                <h2 id="chat-character-name">{activeCharacter.name}</h2>
              </header>
              <div className="chat-messages" id="chat-messages" ref={chatMessagesRef}>
                {messages.map((msg, index) => (
                  <div key={index} className={`message ${msg.sender}`}>
                    {msg.sender === 'bot' ? (
                      <div className={`avatar ${msg.emotion ? msg.emotion.toLowerCase() : ''}`}>
                        {msg.avatarInitial}
                      </div>
                    ) : null}
                    <div className="message-content">{msg.text}</div>
                    {msg.sender === 'user' ? (
                      <div className="avatar">{msg.avatarInitial}</div>
                    ) : null}
                  </div>
                ))}
              </div>
              <div className="chat-input-container">
                <form id="chat-form" className="chat-input-wrapper" onSubmit={handleSendMessage}>
                  <input
                    type="text"
                    id="message-input"
                    placeholder="Ask anything..."
                    value={messageInput}
                    onChange={e => setMessageInput(e.target.value)}
                  />
                  <button id="send-btn" type="submit" title="Send">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                  </button>
                </form>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Command Palette Modal */}
      {isPaletteOpen && (
        <div className="modal-overlay active" id="command-palette-overlay" onClick={() => setPaletteOpen(false)}>
          <div className="modal-content" style={{ maxWidth: '500px' }} onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <input
                type="text"
                id="command-input"
                placeholder="Type a command or search..."
                ref={commandInputRef}
                value={commandInput}
                onChange={e => setCommandInput(e.target.value)}
              />
              <button className="close-btn" onClick={() => setPaletteOpen(false)}>&times;</button>
            </div>
            <div className="modal-body" style={{ paddingTop: '0.5rem' }}>
              <div className="command-results" id="command-results">
                {filteredCommands.length === 0 ? (
                  <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--text-secondary)' }}>No results found.</div>
                ) : (
                  filteredCommands.map(item => (
                    <div
                      key={item.name}
                      className="command-item"
                      onClick={item.handler}
                    >
                      <span style={{ fontSize: '1.25rem' }}>{item.icon}</span>
                      <span>{item.name}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Create Character Modal */}
      {isCreateModalOpen && (
        <div className="modal-overlay active" id="create-character-modal">
          <div className="modal-content">
            <header className="modal-header">
              <h2>Create Character</h2>
              <button className="close-btn" onClick={closeCreateModal}>&times;</button>
            </header>
            <div className="modal-body">
              <div className="creation-instructions">
                <h4>
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                  How to Create a Digital Twin
                </h4>
                <p>To create an accurate digital twin, you need to provide a chat history. The AI will analyze this conversation to learn the character's personality and communication style.</p>
              </div>
              <form id="create-character-form" className="create-character-form" onSubmit={handleCreateCharacter}>
                <div className="form-group">
                  <label htmlFor="char-name">Character Name</label>
                  <input type="text" id="char-name" name="name" required value={createFormState.name} onChange={handleCreateFormChange} />
                </div>
                <div className="form-group">
                  <label htmlFor="char-desc">Description</label>
                  <textarea id="char-desc" name="description" rows="2" required placeholder="A witty and creative AI." value={createFormState.description} onChange={handleCreateFormChange}></textarea>
                </div>
                <div className="form-group">
                  <label htmlFor="my-name">Your Name (in chat file)</label>
                  <input type="text" id="my-name" name="my_name" required value={createFormState.my_name} onChange={handleCreateFormChange} />
                </div>
                <div className="form-group">
                  <label htmlFor="chat-file">Chat History (.txt)</label>
                  <input type="file" id="chat-file" name="file" accept=".txt" required onChange={handleFileChange} />
                </div>
                <button type="submit" className="cta-button" disabled={isCreating}>
                  {isCreating ? 'Creating...' : 'Create Character'}
                </button>
                <div className="form-feedback" id="form-feedback">{formFeedback}</div>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* Character Profile Panel */}
      <aside className={`character-profile-panel ${isProfilePanelOpen ? 'active' : ''}`} id="character-profile-panel">
        {activeCharacter && (
          <>
            <div className="profile-header">
              <button className="close-btn" id="close-panel-btn" onClick={() => setProfilePanelOpen(false)}>&times;</button>
              <div className="profile-avatar">{activeCharacter.name.charAt(0)}</div>
              <h2 className="profile-name">{activeCharacter.name}</h2>
              <p className="profile-description">{activeCharacter.description}</p>
            </div>
            <div className="profile-body">
              <div className="profile-section">
                <h4>Personality Traits</h4>
                <div className="trait-display">
                  <div className="trait"><div className="trait-label"><span>Formal</span><span>{((activeCharacter.traits?.formal || 0) * 100).toFixed(0)}%</span></div><div className="trait-bar"><div className="trait-progress" style={{ width: `${(activeCharacter.traits?.formal || 0) * 100}%` }}></div></div></div>
                  <div className="trait"><div className="trait-label"><span>Casual</span><span>{((activeCharacter.traits?.casual || 0) * 100).toFixed(0)}%</span></div><div className="trait-bar"><div className="trait-progress" style={{ width: `${(activeCharacter.traits?.casual || 0) * 100}%` }}></div></div></div>
                  <div className="trait"><div className="trait-label"><span>Emotional</span><span>{((activeCharacter.traits?.emotional || 0) * 100).toFixed(0)}%</span></div><div className="trait-bar"><div className="trait-progress" style={{ width: `${(activeCharacter.traits?.emotional || 0) * 100}%` }}></div></div></div>
                </div>
              </div>
              <div className="profile-section">
                <h4>Actions</h4>
                <button className="action-button" id="clear-history-btn" onClick={() => handleClearHistory(activeCharacter.name)}>Clear Conversation</button>
                <button className="action-button delete" id="delete-character-btn" onClick={() => handleDeleteCharacter(activeCharacter.name)}>Delete Character</button>
              </div>
            </div>
          </>
        )}
      </aside>

      {/* AI Orb */}
      <div
        className={`ai-orb ${isThinking ? 'thinking' : ''}`}
        id="ai-orb"
        title="Quick Actions (Ctrl+K)"
        onClick={() => setPaletteOpen(true)}
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 3-1.41 1.41L16.17 10H4v4h12.17l-5.58 5.59L12 21l8-8-8-8z" /></svg>
      </div>
    </>
  );
}

export default Dashboard;