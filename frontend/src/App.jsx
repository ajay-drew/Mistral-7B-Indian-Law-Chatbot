import React, { useState, useRef, useEffect } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:2347/chat'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [theme, setTheme] = useState(() => {
    // Check localStorage or default to light
    const savedTheme = localStorage.getItem('theme')
    return savedTheme || 'light'
  })
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

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
    // Apply theme to document root
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    setIsLoading(true)

    // Add user message
    const newUserMessage = { role: 'user', content: userMessage }
    setMessages(prev => [...prev, newUserMessage])

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: userMessage,
          max_new_tokens: 512,
          temperature: 0.7,
          top_p: 0.9
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      const assistantMessage = { role: 'assistant', content: data.reply }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error.message}. Please try again.`
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(e)
    }
  }

  return (
    <div className="app">
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
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <div className="welcome-icon">⚖️</div>
              <h2>Welcome to Mistral Indian Law</h2>
              <p>I'm your specialized AI assistant for Indian legal matters.</p>
              <p>Ask me about constitutional law, criminal law, civil law, or legal procedures.</p>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              <div className="message-content">
                {message.role === 'user' ? (
                  <div className="message-text">{message.content}</div>
                ) : (
                  <div className="message-text">{message.content}</div>
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
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about Indian law..."
              rows={1}
              className="input-field"
              disabled={isLoading}
            />
            <button
              type="submit"
              className="send-button"
              disabled={!input.trim() || isLoading}
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
          <p className="input-hint">Press Enter to send, Shift+Enter for new line</p>
        </form>
      </main>
    </div>
  )
}

export default App

