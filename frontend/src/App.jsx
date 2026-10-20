import { useState, useEffect, useRef } from 'react'
import './App.css'

// Configuration
const API_BASE_URL = 'http://127.0.0.1:8000'
const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']
const SESSION_STORAGE_KEY = 'seefood_prediction_history'
const MAX_HISTORY_ITEMS = 30

/**
 * Helper to safely read initial history from sessionStorage.
 */
function getInitialHistory() {
  try {
    const raw = sessionStorage.getItem(SESSION_STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) {
        return parsed
      }
    }
  } catch (err) {
    console.warn('Failed to load prediction history from sessionStorage:', err)
  }
  return []
}

/**
 * Helper to read a File as a base64 Data URL (for self-contained thumbnails/revisits).
 */
function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

export default function App() {
  // State management
  const [selectedFile, setSelectedFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isDragActive, setIsDragActive] = useState(false)

  // Prediction History state
  const [history, setHistory] = useState(getInitialHistory)
  const [showClearConfirm, setShowClearConfirm] = useState(false)
  const [activeHistoryId, setActiveHistoryId] = useState(null)

  // Backend connection status: 'checking' | 'online' | 'offline'
  const [backendStatus, setBackendStatus] = useState('checking')
  const [backendError, setBackendError] = useState(null)

  // Ref to programmatically open the hidden file input
  const fileInputRef = useRef(null)
  // Ref to track if current imagePreview is an object URL that must be revoked
  const isObjectUrlRef = useRef(false)

  /**
   * Ping GET /health on the FastAPI backend
   */
  const checkHealth = async () => {
    setBackendStatus('checking')
    setBackendError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      })
      if (!response.ok) {
        throw new Error(`Health check returned status ${response.status}`)
      }
      const data = await response.json()
      if (data.status === 'healthy' && data.model_loaded) {
        setBackendStatus('online')
      } else {
        setBackendStatus('offline')
        setBackendError('Backend reached, but the model is not loaded.')
      }
    } catch {
      setBackendStatus('offline')
      setBackendError('Cannot connect to FastAPI backend at http://127.0.0.1:8000. Is the server running?')
    }
  }

  // Check backend health on component mount
  useEffect(() => {
    let ignore = false
    async function runCheck() {
      try {
        const response = await fetch(`${API_BASE_URL}/health`, {
          method: 'GET',
          headers: { Accept: 'application/json' },
        })
        if (!response.ok) {
          throw new Error(`Health check status ${response.status}`)
        }
        const data = await response.json()
        if (!ignore) {
          if (data.status === 'healthy' && data.model_loaded) {
            setBackendStatus('online')
          } else {
            setBackendStatus('offline')
            setBackendError('Backend reached, but the model is not loaded.')
          }
        }
      } catch {
        if (!ignore) {
          setBackendStatus('offline')
          setBackendError('Cannot connect to FastAPI backend at http://127.0.0.1:8000. Is the server running?')
        }
      }
    }
    runCheck()
    return () => {
      ignore = true
    }
  }, [])

  // Sync history state to sessionStorage whenever it updates
  useEffect(() => {
    try {
      sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(history))
    } catch (err) {
      console.warn('Unable to persist history to sessionStorage (quota exceeded?):', err)
    }
  }, [history])

  // Revoke object URL when imagePreview changes or unmounts to prevent memory leaks
  useEffect(() => {
    const currentUrl = imagePreview
    const wasObjectUrl = isObjectUrlRef.current
    return () => {
      if (wasObjectUrl && currentUrl && currentUrl.startsWith('blob:')) {
        URL.revokeObjectURL(currentUrl)
      }
    }
  }, [imagePreview])

  /**
   * Validate image file type and extension
   */
  const validateFile = (file) => {
    if (!file) return false

    const fileName = file.name.toLowerCase()
    const hasValidExt = ALLOWED_EXTENSIONS.some((ext) => fileName.endsWith(ext))
    const hasValidMime = ALLOWED_MIME_TYPES.includes(file.type)

    if (!hasValidExt && !hasValidMime) {
      setError(
        `Unsupported file type "${file.name}". Please select a JPEG, PNG, or WebP image.`
      )
      return false
    }

    return true
  }

  /**
   * Process a valid file for preview and upload
   */
  const handleFileSelect = (file) => {
    setError(null)
    setPrediction(null)
    setActiveHistoryId(null)

    if (!validateFile(file)) {
      return
    }

    // Clean up previous blob URL if needed
    if (isObjectUrlRef.current && imagePreview && imagePreview.startsWith('blob:')) {
      URL.revokeObjectURL(imagePreview)
    }

    setSelectedFile(file)
    const previewUrl = URL.createObjectURL(file)
    isObjectUrlRef.current = true
    setImagePreview(previewUrl)
  }

  // Handle standard file input change
  const handleInputChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  // Drag and drop event handlers
  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)

    const file = e.dataTransfer.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  /**
   * Send the image to FastAPI POST /predict
   */
  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select or drop an image first.')
      return
    }

    setLoading(true)
    setError(null)
    setPrediction(null)

    const currentFile = selectedFile
    const formData = new FormData()
    formData.append('file', currentFile)

    try {
      // Convert file to data URL in parallel for durable history storage
      const [response, dataUrl] = await Promise.all([
        fetch(`${API_BASE_URL}/predict`, {
          method: 'POST',
          body: formData,
        }),
        fileToDataUrl(currentFile),
      ])

      if (!response.ok) {
        let errorDetail = `Classification failed with status ${response.status}.`
        try {
          const errorJson = await response.json()
          if (errorJson.detail) {
            errorDetail = errorJson.detail
          }
        } catch {
          // If response body is not JSON, retain default errorDetail
        }
        throw new Error(errorDetail)
      }

      const data = await response.json()
      setPrediction(data)

      // Record in history without duplicate triggers
      const historyEntryId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
      const newEntry = {
        id: historyEntryId,
        fileName: currentFile.name,
        fileSize: currentFile.size,
        timestamp: new Date().toISOString(),
        imageDataUrl: dataUrl,
        prediction: {
          label: data.label,
          hotdog_probability: data.hotdog_probability,
          confidence: data.confidence,
          gradcam_base64: data.gradcam_base64,
        },
      }

      setActiveHistoryId(historyEntryId)
      setHistory((prev) => [newEntry, ...prev].slice(0, MAX_HISTORY_ITEMS))
    } catch (err) {
      console.error('Prediction error:', err)
      setError(err.message || 'Network error: could not connect to backend.')
    } finally {
      setLoading(false)
    }
  }

  /**
   * Revisit a past history entry without re-querying the backend
   */
  const handleSelectHistoryItem = (entry) => {
    // Revoke any active blob URL before switching to history's data URL
    if (isObjectUrlRef.current && imagePreview && imagePreview.startsWith('blob:')) {
      URL.revokeObjectURL(imagePreview)
    }
    isObjectUrlRef.current = false
    setSelectedFile(null)
    setImagePreview(entry.imageDataUrl)
    setPrediction(entry.prediction)
    setActiveHistoryId(entry.id)
    setError(null)

    // Smoothly scroll back to the result or main classifier section
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  /**
   * Clear History with confirmation
   */
  const handleConfirmClearHistory = () => {
    setHistory([])
    setShowClearConfirm(false)
    try {
      sessionStorage.removeItem(SESSION_STORAGE_KEY)
    } catch (err) {
      console.warn('Failed to clear sessionStorage:', err)
    }
  }

  /**
   * Reset everything to start fresh
   */
  const handleReset = () => {
    if (isObjectUrlRef.current && imagePreview && imagePreview.startsWith('blob:')) {
      URL.revokeObjectURL(imagePreview)
    }
    isObjectUrlRef.current = false
    setSelectedFile(null)
    setImagePreview(null)
    setPrediction(null)
    setActiveHistoryId(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Safe formatting helpers
  const formatConfidence = (val) => {
    if (typeof val === 'number' && !isNaN(val)) {
      return `${val.toFixed(1)}%`
    }
    return null
  }

  const formatProbability = (val) => {
    if (typeof val === 'number' && !isNaN(val)) {
      return `${(val * 100).toFixed(1)}%`
    }
    return 'N/A'
  }

  const formatTimestamp = (isoString) => {
    if (!isoString) return ''
    try {
      const d = new Date(isoString)
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    } catch {
      return ''
    }
  }

  const labelLower = prediction?.label?.toLowerCase() || ''
  const isHotdog = labelLower === 'hotdog'
  const isUncertain = labelLower === 'uncertain'
  const isNotHotdog = labelLower === 'not hotdog'

  // Validate Grad-CAM data URL safely
  const hasValidGradcam =
    typeof prediction?.gradcam_base64 === 'string' &&
    prediction.gradcam_base64.startsWith('data:image/jpeg;base64,')

  return (
    <main className="app-container">
      {/* Header with Title and Backend Health Status */}
      <header className="header-section">
        <div
          className={`status-badge ${backendStatus}`}
          title={backendError || (backendStatus === 'online' ? 'Backend connected & model ready' : 'Checking status')}
        >
          <span className="status-dot" aria-hidden="true"></span>
          <span>
            {backendStatus === 'online' && 'Backend Connected'}
            {backendStatus === 'offline' && 'Backend Offline'}
            {backendStatus === 'checking' && 'Checking Backend...'}
          </span>
          {backendStatus === 'offline' && (
            <button
              onClick={checkHealth}
              className="browse-btn"
              style={{ padding: '0.1rem 0.4rem', fontSize: '0.75rem', marginLeft: '0.25rem' }}
              aria-label="Retry backend connection"
            >
              Retry
            </button>
          )}
        </div>

        <h1 className="brand-title">
          <span className="brand-emoji" aria-hidden="true">🌭</span>
          SeeFood
        </h1>
        <p className="brand-subtitle">
          The ultimate AI food classification engine. Instant verification: is it a hotdog, or not a hotdog?
        </p>
      </header>

      {/* Main Glass Card */}
      <section className="card" aria-label="Image Classifier Section">
        {/* Upload Zone / Drag & Drop */}
        {!imagePreview ? (
          <div
            className={`dropzone ${isDragActive ? 'drag-active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                fileInputRef.current?.click()
              }
            }}
            aria-label="Upload image zone. Drag and drop an image or click to browse"
          >
            <div className="upload-icon-container" aria-hidden="true">
              📷
            </div>
            <div className="dropzone-text">
              <h3>Drag and drop your food photo here</h3>
              <p>Supports JPEG, PNG, and WebP (up to 10MB)</p>
            </div>
            <button
              type="button"
              className="browse-btn"
              onClick={(e) => {
                e.stopPropagation()
                fileInputRef.current?.click()
              }}
            >
              Browse Image
            </button>
          </div>
        ) : (
          /* Preview Selected Image */
          <div className="preview-container">
            <div className="preview-wrapper">
              <img
                src={imagePreview}
                alt="Selected food preview"
                className="image-preview"
              />
            </div>
            <div className="preview-file-info">
              {selectedFile ? (
                <>
                  <span>{selectedFile.name}</span>
                  <span>•</span>
                  <span>{(selectedFile.size / 1024).toFixed(1)} KB</span>
                </>
              ) : (
                <span className="history-loaded-tag">
                  Viewing prediction from history
                </span>
              )}
            </div>
          </div>
        )}

        {/* Hidden File Input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
          onChange={handleInputChange}
          style={{ display: 'none' }}
          aria-hidden="true"
        />

        {/* Action Controls */}
        <div className="button-row">
          <button
            type="button"
            className="btn btn-primary"
            onClick={handlePredict}
            disabled={!selectedFile || loading}
            aria-busy={loading}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true"></span>
                <span>Analyzing Image & Explainability...</span>
              </>
            ) : (
              <>
                <span>⚡</span>
                <span>Classify Image</span>
              </>
            )}
          </button>

          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleReset}
            disabled={loading || (!selectedFile && !prediction && !error && !imagePreview)}
          >
            Reset
          </button>
        </div>

        {/* Error Notification */}
        {error && (
          <div className="error-banner" role="alert">
            <span className="error-icon" aria-hidden="true">⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* Offline Backend Notice if offline */}
        {backendStatus === 'offline' && backendError && !error && (
          <div className="error-banner" role="alert">
            <span className="error-icon" aria-hidden="true">🔌</span>
            <span>{backendError}</span>
          </div>
        )}

        {/* Prediction Results & Grad-CAM Visual Explainability */}
        {prediction && (
          <article
            className={`result-card ${
              isHotdog ? 'hotdog' : isUncertain ? 'uncertain' : 'not-hotdog'
            }`}
            aria-live="polite"
          >
            <div className="result-header">
              <span className="result-icon" aria-hidden="true">
                {isHotdog ? '🌭' : isUncertain ? '🤔' : '🚫'}
              </span>
              <div>
                <h2 className="result-title">
                  {prediction.label || 'Unknown'}
                </h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  {isHotdog && 'AI verified! This item is classified as a Hotdog.'}
                  {isNotHotdog && 'AI verified! This item is classified as NOT a Hotdog.'}
                  {isUncertain &&
                    'The model is unsure about this image. Try a clearer photo or review the result cautiously.'}
                </p>
              </div>
            </div>

            <div className="metrics-grid">
              <div className="metric-item">
                <span className="metric-label">Prediction Confidence</span>
                <span className="metric-value">
                  {isUncertain
                    ? 'Indeterminate'
                    : formatConfidence(prediction.confidence) || 'N/A'}
                </span>
                <div className="progress-bar-container">
                  <div
                    className="progress-fill"
                    style={{
                      width: isUncertain
                        ? '0%'
                        : `${Math.min(100, Math.max(0, prediction.confidence || 0))}%`,
                    }}
                  ></div>
                </div>
              </div>

              <div className="metric-item">
                <span className="metric-label">Hotdog Probability</span>
                <span className="metric-value">
                  {formatProbability(prediction.hotdog_probability)}
                </span>
                <div className="progress-bar-container">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${Math.min(100, Math.max(0, (prediction.hotdog_probability || 0) * 100))}%`,
                    }}
                  ></div>
                </div>
              </div>
            </div>

            <p className="provisional-note">
              Decision thresholds are provisional until calibrated on validation data.
            </p>

            {/* Grad-CAM Explainability Section */}
            <div className="explain-section">
              <div className="explain-header">
                <h3>
                  <span aria-hidden="true">🔍</span>
                  Where the AI Looked
                </h3>
                <p className="explain-disclaimer">
                  This Grad-CAM heatmap highlights the key visual regions and patterns that most heavily
                  influenced the neural network's prediction. It highlights model focus areas, but is not an
                  absolute guarantee of classification accuracy.
                </p>
              </div>

              {hasValidGradcam ? (
                <div className="comparison-grid">
                  <div className="comparison-column">
                    <span className="comparison-label">Original Photo</span>
                    <div className="comparison-media-wrapper">
                      <img
                        src={imagePreview}
                        alt="Original food photo uploaded by user"
                        className="comparison-image"
                      />
                    </div>
                  </div>

                  <div className="comparison-column">
                    <span className="comparison-label">Grad-CAM Heatmap Focus</span>
                    <div className="comparison-media-wrapper">
                      <img
                        src={prediction.gradcam_base64}
                        alt="Grad-CAM visual heatmap overlay showing areas that influenced the prediction"
                        className="comparison-image"
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="explain-unavailable">
                  Visual explanation is currently unavailable for this prediction.
                </div>
              )}
            </div>
          </article>
        )}
      </section>

      {/* Prediction History Section */}
      <section className="history-section card" aria-label="Prediction History">
        <div className="history-header">
          <div className="history-title-group">
            <h2 className="history-title">
              <span aria-hidden="true">📜</span>
              Prediction History
            </h2>
            <span className="history-count-badge">
              {history.length} {history.length === 1 ? 'entry' : 'entries'}
            </span>
          </div>

          {history.length > 0 && (
            <div className="history-actions">
              {!showClearConfirm ? (
                <button
                  type="button"
                  className="btn-history-clear"
                  onClick={() => setShowClearConfirm(true)}
                  aria-label="Clear all prediction history"
                >
                  Clear History
                </button>
              ) : (
                <div className="clear-confirm-group" role="group" aria-label="Confirm clear history">
                  <span className="clear-confirm-text">Clear all {history.length} entries?</span>
                  <button
                    type="button"
                    className="btn-confirm-yes"
                    onClick={handleConfirmClearHistory}
                  >
                    Yes, Clear
                  </button>
                  <button
                    type="button"
                    className="btn-confirm-cancel"
                    onClick={() => setShowClearConfirm(false)}
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {history.length === 0 ? (
          <div className="history-empty">
            <span className="history-empty-icon" aria-hidden="true">🍽️</span>
            <p>No predictions yet in this session.</p>
            <span className="history-empty-hint">
              Images you classify will appear here so you can revisit their details and Grad-CAM maps anytime.
            </span>
          </div>
        ) : (
          <div className="history-list" role="feed" aria-label="Previous predictions list">
            {history.map((item) => {
              const itemLabel = item.prediction?.label?.toLowerCase() || ''
              const itemIsHotdog = itemLabel === 'hotdog'
              const itemIsUncertain = itemLabel === 'uncertain'
              const isActive = item.id === activeHistoryId
              const itemConfidence = formatConfidence(item.prediction?.confidence)

              return (
                <article
                  key={item.id}
                  className={`history-item ${
                    itemIsHotdog ? 'item-hotdog' : itemIsUncertain ? 'item-uncertain' : 'item-not-hotdog'
                  } ${isActive ? 'item-active' : ''}`}
                  onClick={() => handleSelectHistoryItem(item)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      handleSelectHistoryItem(item)
                    }
                  }}
                  aria-label={`Revisit prediction for ${item.fileName || 'image'}: ${item.prediction?.label}`}
                >
                  <div className="history-thumb-wrapper">
                    <img
                      src={item.imageDataUrl}
                      alt={`Thumbnail of ${item.fileName || 'food'}`}
                      className="history-thumb-img"
                    />
                    <span className="history-thumb-badge" aria-hidden="true">
                      {itemIsHotdog ? '🌭' : itemIsUncertain ? '🤔' : '🚫'}
                    </span>
                  </div>

                  <div className="history-details">
                    <div className="history-details-top">
                      <span className="history-item-label">
                        {item.prediction?.label}
                      </span>
                      <time className="history-item-time" dateTime={item.timestamp}>
                        {formatTimestamp(item.timestamp)}
                      </time>
                    </div>

                    <div className="history-item-stats">
                      <span className="history-stat">
                        <strong className="history-stat-name">Prob:</strong>{' '}
                        {formatProbability(item.prediction?.hotdog_probability)}
                      </span>
                      {/* Only display confidence if it is numeric, never invent one for Uncertain */}
                      {itemConfidence && (
                        <span className="history-stat">
                          <strong className="history-stat-name">Conf:</strong>{' '}
                          {itemConfidence}
                        </span>
                      )}
                      {itemIsUncertain && (
                        <span className="history-stat uncertain-badge">
                          Uncertain
                        </span>
                      )}
                    </div>
                  </div>

                  <span className="history-view-arrow" aria-hidden="true">
                    {isActive ? '●' : '→'}
                  </span>
                </article>
              )
            })}
          </div>
        )}
      </section>
    </main>
  )
}

