import React, { useState, useRef, useEffect } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL?.replace('/chat', '') || 'http://localhost:2347'
const CHAT_API_URL = `${API_BASE}/chat`
const CHAT_STREAM_API_URL = `${API_BASE}/chat/stream`
const DOCUMENTS_API_URL = `${API_BASE}/documents`
const HEALTH_API_URL = `${API_BASE}/health`
const API_KEY = import.meta.env.VITE_API_KEY || null
const MAX_INPUT_LENGTH = 1000
const REQUEST_TIMEOUT = 60000 // 60 seconds
const HEALTH_CHECK_INTERVAL = 30000 // 30 seconds

// Utility function for user-friendly error messages
const getUserFriendlyError = (error) => {
  const msg = error.message || String(error)
  if (msg.includes('timeout') || msg.includes('AbortError')) {
    return 'The request took too long. Please try again with a shorter question.'
  }
  if (msg.includes('Failed to upload') || msg.includes('upload')) {
    return 'The document could not be uploaded. Please check if it\'s a valid PDF file (max 10MB).'
  }
  if (msg.includes('Failed to fetch') || msg.includes('network') || msg.includes('NetworkError')) {
    return 'Cannot connect to server. Please check your internet connection.'
  }
  if (msg.includes('500') || msg.includes('Internal')) {
    return 'The server encountered an error. Please try again later.'
  }
  return 'Something went wrong. Please try again.'
}

// Fetch with timeout wrapper
const fetchWithTimeout = async (url, options = {}, timeout = REQUEST_TIMEOUT) => {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    })
    clearTimeout(timeoutId)
    return response
  } catch (error) {
    clearTimeout(timeoutId)
    if (error.name === 'AbortError') {
      throw new Error('Request timed out. Please try again.')
    }
    throw error
  }
}

function App() {
  const [messages, setMessages] = useState(() => {
    // Load messages from localStorage on mount
    try {
      const saved = localStorage.getItem('chat_messages')
      if (saved) {
        return JSON.parse(saved)
      }
    } catch {
      // Ignore localStorage errors
    }
    return []
  })
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [documents, setDocuments] = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [uploadStage, setUploadStage] = useState('idle')
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [connectionError, setConnectionError] = useState(null)
  const [documentsError, setDocumentsError] = useState(null)
  const [loadingDocuments, setLoadingDocuments] = useState(true)
  const [inputError, setInputError] = useState('')
  const [theme, setTheme] = useState(() => {
    try {
      const savedTheme = localStorage.getItem('theme')
      return savedTheme || 'light'
    } catch {
      return 'light'
    }
  })
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const fileInputRef = useRef(null)

  const hasDocuments = documents.length > 0

  // Save messages to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('chat_messages', JSON.stringify(messages))
    } catch {
      // Ignore localStorage errors
    }
  }, [messages])

  // Health check on mount and periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetchWithTimeout(HEALTH_API_URL, {}, 5000)
        if (response.ok) {
          setConnectionError(null)
        } else {
          setConnectionError('Backend is not available')
        }
      } catch (error) {
        setConnectionError('Cannot connect to server')
      }
    }

    // Check immediately
    checkHealth()

    // Check periodically
    const interval = setInterval(checkHealth, HEALTH_CHECK_INTERVAL)
    return () => clearInterval(interval)
  }, [])

  // Input validation
  const validateInput = (text) => {
    if (!text.trim()) {
      setInputError('Please enter a question')
      return false
    }
    if (text.length > MAX_INPUT_LENGTH) {
      setInputError(`Question is too long (max ${MAX_INPUT_LENGTH} characters)`)
      return false
    }
    setInputError('')
    return true
  }

  // Fetch documents on mount
  useEffect(() => {
    fetchDocuments()
  }, [])

  const fetchDocuments = async () => {
    setLoadingDocuments(true)
    setDocumentsError(null)
    try {
      const response = await fetchWithTimeout(DOCUMENTS_API_URL)
      if (response.ok) {
        const data = await response.json()
        setDocuments(data.documents || [])
        setDocumentsError(null)
      } else {
        const errorMsg = getUserFriendlyError(new Error(`Failed to fetch documents: ${response.status}`))
        setDocumentsError(errorMsg)
      }
    } catch (error) {
      console.error('Error fetching documents:', error)
      setDocumentsError(getUserFriendlyError(error))
    } finally {
      setLoadingDocuments(false)
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem('theme', theme)
    } catch {}
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Please upload a PDF file')
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      setUploadError('File size must be less than 10MB')
      return
    }

    setUploading(true)
    setUploadError(null)
    setUploadSuccess(false)
    setUploadStage('uploading')

    try {
      const formData = new FormData()
      formData.append('file', file)

      setUploadStage('processing')
      const response = await fetchWithTimeout(`${DOCUMENTS_API_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      setUploadStage('indexing')
      const data = await response.json()
      await fetchDocuments()
      setUploadError(null)
      setUploadStage('complete')
      setUploadSuccess(true)
      
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }

      // Auto-dismiss success message after 3 seconds
      setTimeout(() => {
        setUploadSuccess(false)
        setUploadStage('idle')
      }, 3000)
    } catch (error) {
      console.error('Upload error:', error)
      setUploadError(getUserFriendlyError(error))
      setUploadStage('idle')
    } finally {
      setUploading(false)
    }
  }

  const handleDeleteDocument = async (docId) => {
    // Confirmation dialog
    if (!window.confirm('Are you sure you want to delete this document?')) {
      return
    }

    try {
      const response = await fetchWithTimeout(`${DOCUMENTS_API_URL}/${docId}`, {
        method: 'DELETE',
      })

      if (!response.ok) {
        throw new Error('Failed to delete document')
      }

      await fetchDocuments()
    } catch (error) {
      console.error('Delete error:', error)
      alert(getUserFriendlyError(error))
    }
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    // Validate input
    if (!validateInput(input)) {
      return
    }

    const userMessage = input.trim()
    setInput('')
    setInputError('')
    setIsLoading(true)

    const newUserMessage = { role: 'user', content: userMessage }
    setMessages(prev => [...prev, newUserMessage])

    // Add placeholder for streaming response
    const assistantMessageId = Date.now()
    const assistantMessage = { 
      role: 'assistant', 
      content: '',
      sources: null,
      id: assistantMessageId
    }
    setMessages(prev => [...prev, assistantMessage])

    try {
      // Try streaming first, fallback to regular endpoint
      const useStreaming = true
      
      if (useStreaming) {
        await sendMessageStream(userMessage, assistantMessageId)
      } else {
        await sendMessageRegular(userMessage, assistantMessageId)
      }
    } catch (error) {
      console.error('Error:', error)
      // Update the assistant message with error
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessageId 
          ? { ...msg, content: `Sorry, I encountered an error: ${getUserFriendlyError(error)}. Please try again.` }
          : msg
      ))
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }

  const sendMessageStream = async (userMessage, assistantMessageId) => {
    const headers = {
      'Content-Type': 'application/json',
    }
    if (API_KEY) {
      headers['X-API-Key'] = API_KEY
    }

    const response = await fetch(CHAT_STREAM_API_URL, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({
        prompt: userMessage,
        max_new_tokens: 512,
        temperature: 0.7,
        top_p: 0.9
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Request failed' }))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let fullContent = ''
    let sources = null
    
    // Batch state updates for better performance (update every 50ms or when batch fills)
    let updateBuffer = ''
    let lastUpdateTime = Date.now()
    const UPDATE_INTERVAL = 50 // ms
    const MAX_BUFFER_SIZE = 20 // characters before forcing update
    
    const flushUpdate = () => {
      if (updateBuffer) {
        fullContent += updateBuffer
        setMessages(prev => prev.map(msg => 
          msg.id === assistantMessageId 
            ? { ...msg, content: fullContent }
            : msg
        ))
        updateBuffer = ''
        lastUpdateTime = Date.now()
      }
    }
    
    // Use requestAnimationFrame for smooth updates
    let rafId = null
    const scheduleUpdate = () => {
      if (rafId) return // Already scheduled
      rafId = requestAnimationFrame(() => {
        flushUpdate()
        rafId = null
      })
    }

    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        // Flush any remaining buffer
        flushUpdate()
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim()
          if (data === '' || data === '[DONE]') continue

          try {
            const parsed = JSON.parse(data)
            
            if (parsed.token) {
              // Add to update buffer (batched tokens from backend)
              updateBuffer += parsed.token
              
              // Schedule update if buffer is large enough or timeout reached
              const timeSinceUpdate = Date.now() - lastUpdateTime
              if (updateBuffer.length >= MAX_BUFFER_SIZE || timeSinceUpdate >= UPDATE_INTERVAL) {
                flushUpdate()
              } else {
                scheduleUpdate()
              }
            }
            
            if (parsed.sources) {
              sources = parsed.sources
            }
            
            if (parsed.done) {
              // Final flush and update with sources
              flushUpdate()
              setMessages(prev => prev.map(msg => 
                msg.id === assistantMessageId 
                  ? { ...msg, content: fullContent, sources: sources }
                  : msg
              ))
              return
            }
            
            if (parsed.error) {
              throw new Error(parsed.error)
            }
          } catch (e) {
            if (e instanceof SyntaxError) {
              // Skip malformed JSON
              continue
            }
            throw e
          }
        }
      }
    }
  }

  const sendMessageRegular = async (userMessage, assistantMessageId) => {
    const headers = {
      'Content-Type': 'application/json',
    }
    if (API_KEY) {
      headers['X-API-Key'] = API_KEY
    }

    const response = await fetchWithTimeout(CHAT_API_URL, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({
        prompt: userMessage,
        max_new_tokens: 512,
        temperature: 0.7,
        top_p: 0.9
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Request failed' }))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    setMessages(prev => prev.map(msg => 
      msg.id === assistantMessageId 
        ? { ...msg, content: data.reply, sources: data.sources || null }
        : msg
    ))
  }

  const clearConversation = () => {
    if (window.confirm('Are you sure you want to clear the conversation?')) {
      setMessages([])
      try {
        localStorage.removeItem('chat_messages')
      } catch {
        // Ignore localStorage errors
      }
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(e)
    }
  }

  const handleInputChange = (e) => {
    const value = e.target.value
    setInput(value)
    // Real-time validation
    if (value.length > MAX_INPUT_LENGTH) {
      setInputError(`Question is too long (max ${MAX_INPUT_LENGTH} characters)`)
    } else if (value.trim() && inputError) {
      setInputError('')
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString)
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } catch {
      return dateString
    }
  }

  return (
    <div className="app">
      {/* Connection Status Banner */}
      {connectionError && (
        <div className="connection-error-banner">
          <span>⚠️ {connectionError}</span>
        </div>
      )}

      <div className="created-by">
        <div className="created-by-content">
          <span className="created-by-label">Created by</span>
          <a href="https://www.linkedin.com/in/ajay-drew/" target="_blank" rel="noopener noreferrer" className="created-by-name">
            Ajay A
          </a>
          <a href="mailto:drewjay05@gmail.com" className="created-by-email">
            drewjay05@gmail.com
          </a>
        </div>
      </div>
      
      <header className="app-header">
        <div className="header-content">
          <h1>Mistral Indian Law</h1>
          <p className="subtitle">Your AI Assistant for Indian Legal Matters</p>
          {hasDocuments && (
            <span className="document-badge">{documents.length} document{documents.length !== 1 ? 's' : ''} loaded</span>
          )}
        </div>
        <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
          {theme === 'light' ? (
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M10 3V1M10 19V17M17 10H19M1 10H3M15.657 15.657L16.97 16.97M3.343 3.343L4.657 4.657M15.657 4.343L16.97 3.03M3.343 16.657L4.657 15.343M13 10C13 11.6569 11.6569 13 10 13C8.34315 13 7 11.6569 7 10C7 8.34315 8.34315 7 10 7C11.6569 7 13 8.34315 13 10Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M17.293 13.293C16.3785 14.2075 15.2348 14.8621 13.9954 15.2009C12.756 15.5397 11.4604 15.5532 10.2146 15.2404C8.96879 14.9276 7.80947 14.2975 6.88484 13.3729C5.96021 12.4483 5.33007 11.2889 5.01729 10.0431C4.7045 8.79729 4.71798 7.50171 5.05677 6.26229C5.39557 5.02287 6.05018 3.87918 6.96469 2.96469C7.8792 2.05018 9.02289 1.39557 10.2623 1.05677C11.5017 0.717975 12.7973 0.704495 14.0431 1.01728C15.2889 1.33007 16.4483 1.96021 17.3729 2.88484C18.2975 3.80947 18.9276 4.96879 19.2404 6.21459C19.5532 7.46039 19.5397 8.75597 19.2009 9.99539C18.8621 11.2348 18.2075 12.3785 17.293 13.293Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )}
        </button>
      </header>

      <main className="chat-container">
        {/* Hidden file input for RAG upload via attach button */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileUpload}
          disabled={uploading}
          style={{ display: 'none' }}
          id="file-upload"
        />

        {/* Chat Section */}
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <div className="welcome-icon">⚖️</div>
              <h2>Welcome to Mistral Indian Law</h2>
              <p>I'm your specialized AI assistant for Indian legal matters.</p>
              {hasDocuments ? (
                <p>Ask me questions about Indian law. I can also answer based on your uploaded documents using RAG.</p>
              ) : (
                <p>Ask me questions about Indian law. You can also upload PDF documents using the attach button to enhance my responses with document-specific information.</p>
              )}
            </div>
          )}

          {messages.length > 0 && (
            <div className="conversation-header">
              <button onClick={clearConversation} className="clear-conversation-button">
                Clear Conversation
              </button>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              <div className="message-content">
                <div className="message-text">{message.content}</div>
                {message.sources && message.sources.length > 0 && (
                  <div className="message-sources">
                    <strong>Based on:</strong> {message.sources.join(', ')}
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message assistant-message">
              <div className="message-content">
                <div className="loading-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form className="input-container" onSubmit={sendMessage}>
          <div className="input-wrapper">
            <button
              type="button"
              className="attach-button"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              aria-label={uploading ? "Uploading document..." : "Attach PDF document"}
              title={uploading ? "Uploading..." : "Attach PDF document"}
            >
              {uploading ? (
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 20 20"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  className="spinning"
                >
                  <circle
                    cx="10"
                    cy="10"
                    r="8"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeDasharray="31.416"
                    strokeDashoffset="23.562"
                    opacity="0.3"
                  />
                  <circle
                    cx="10"
                    cy="10"
                    r="8"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeDasharray="31.416"
                    strokeDashoffset="15.708"
                  />
                </svg>
              ) : (
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 20 20"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M12.5 5.5V14.5C12.5 16.1569 11.1569 17.5 9.5 17.5C7.84315 17.5 6.5 16.1569 6.5 14.5V4.5C6.5 3.39543 7.39543 2.5 8.5 2.5C9.60457 2.5 10.5 3.39543 10.5 4.5V13.5C10.5 13.7761 10.2761 14 10 14C9.72386 14 9.5 13.7761 9.5 13.5V5.5"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </button>
            <textarea
              ref={inputRef}
              value={input}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              placeholder="Ask about Indian law..."
              rows={1}
              className="input-field"
              disabled={isLoading}
              maxLength={MAX_INPUT_LENGTH}
            />
            <button
              type="submit"
              className="send-button"
              disabled={!input.trim() || isLoading || !!inputError}
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 20 20"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M18 2L9 11M18 2L12 18L9 11M18 2L2 8L9 11"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
          <div className="input-hint-container">
            {uploading && (
              <p className="upload-status">
                {uploadStage === 'uploading' ? '📤 Uploading...' :
                 uploadStage === 'processing' ? '⚙️ Processing...' :
                 uploadStage === 'indexing' ? '🔍 Indexing...' :
                 '📤 Uploading...'}
              </p>
            )}
            {uploadError && (
              <p className="upload-error-hint">{uploadError}</p>
            )}
            {uploadSuccess && (
              <p className="upload-success-hint">✓ Document uploaded successfully!</p>
            )}
            {inputError && (
              <p className="input-error-hint">{inputError}</p>
            )}
            <p className="input-hint">
              {input.length}/{MAX_INPUT_LENGTH} characters • Press Enter to send, Shift+Enter for new line
              {hasDocuments && " • RAG enabled with uploaded documents"}
            </p>
          </div>
        </form>
      </main>
    </div>
  )
}

export default App
