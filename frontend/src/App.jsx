
import { useState } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Loader2,
  Send,
  Sparkles,
  Smile,
  Frown,
  Meh,
  Zap,
  Activity,
  Heart,
  Ghost
} from 'lucide-react'
import './App.css'

function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!text.trim()) {
      setError("Please enter some text")
      return;
    }
    setError(null)
    setResult(null)
    setLoading(true)

    try {
      // Direct call to Flask backend exposed by CORS
      const response = await axios.post('http://127.0.0.1:5000/predict', {
        text: text
      })

      // Artificial delay for smooth UX if too fast
      // await new Promise(r => setTimeout(r, 600)); 

      setResult(response.data)
    } catch (err) {
      console.error(err)
      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error)
      } else {
        setError("Could not connect to the AI model. Ensure the backend server is running.")
      }
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  // Animation Variants
  const containerVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: "spring",
        stiffness: 100,
        damping: 15,
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  }

  return (
    <div className="app-container">
      {/* Dynamic Background Elements */}
      <div className="bg-shape shape-1"></div>
      <div className="bg-shape shape-2"></div>

      <motion.div
        className="card glass-card"
        initial="hidden"
        animate="visible"
        variants={containerVariants}
      >
        <motion.div variants={itemVariants} className="header">
          <div className="icon-wrapper">
            <Sparkles size={32} className="text-primary" />
          </div>
          <h1>Tanglish AI</h1>
        </motion.div>

        <motion.p variants={itemVariants} className="subtitle">
          Advanced Sentiment & Emotion Detection engine for Tamil-English code-mixed text.
        </motion.p>

        <form onSubmit={handleSubmit}>
          <motion.div variants={itemVariants} className="input-group">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Type your Tanglish text here... (e.g., 'Padam semma mass but story konjam lag')"
              rows="4"
              className="glass-input"
            />
          </motion.div>

          <motion.button
            variants={itemVariants}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            type="submit"
            disabled={loading}
            className="submit-btn"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin" size={20} /> Analyzing...
              </>
            ) : (
              <>
                Analyze Text <Send size={18} />
              </>
            )}
          </motion.button>
        </form>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="error-msg"
            >
              <Zap size={18} /> {error}
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {result && (
            <motion.div
              className="results-container"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ type: "spring", stiffness: 200, damping: 20 }}
            >
              <ResultCard
                title="Sentiment"
                value={result.sentiment}
                confidence={result.sentiment_confidence}
                type="sentiment"
              />

              <ResultCard
                title="Emotion"
                value={result.emotion}
                confidence={result.emotion_confidence}
                type="emotion"
              />

              {/* AI Insight Section */}
              <motion.div
                className="insight-card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <div className="insight-header">
                  <Activity className="text-secondary" size={20} />
                  <span>AI Insight</span>
                </div>
                <p className="insight-text">
                  {result.decision || "Analyzing context..."}
                </p>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}

// Component for Individual Result Cards
const ResultCard = ({ title, value, confidence, type }) => {
  const getIcon = () => {
    if (type === 'sentiment') {
      if (value.toLowerCase().includes('positive')) return <Smile size={24} className="text-green-400" />
      if (value.toLowerCase().includes('negative')) return <Frown size={24} className="text-red-400" />
      return <Meh size={24} className="text-yellow-400" />
    } else {
      if (['joy', 'love', 'happy'].some(x => value.toLowerCase().includes(x))) return <Heart size={24} className="text-pink-400" />
      if (['fear', 'sadness'].some(x => value.toLowerCase().includes(x))) return <Ghost size={24} className="text-blue-400" />
      return <Zap size={24} className="text-purple-400" />
    }
  }

  const getColor = () => {
    if (type === 'sentiment') return getSentimentColor(value)
    return getEmotionColor(value)
  }

  return (
    <motion.div
      className="result-card glass-panel"
      whileHover={{ y: -5, boxShadow: "0 10px 30px -10px rgba(0,0,0,0.5)" }}
    >
      <div className="result-header">
        <span className="result-label">{title}</span>
        {getIcon()}
      </div>
      <div className="result-value" style={{ color: getColor() }}>
        {value}
      </div>
      <div className="confidence-wrapper">
        <div className="confidence-bar-bg">
          <motion.div
            className="confidence-bar-fill"
            initial={{ width: 0 }}
            animate={{ width: `${confidence}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
            style={{ backgroundColor: getColor() }}
          />
        </div>
        <span className="confidence-text">{confidence}% Confidence</span>
      </div>
    </motion.div>
  )
}

// Color Helpers
const getSentimentColor = (s) => {
  const sentiment = s?.toLowerCase() || '';
  if (sentiment.includes('positive')) return '#4ade80';
  if (sentiment.includes('negative')) return '#f87171';
  if (sentiment.includes('mixed')) return '#fbbf24';
  return '#94a3b8';
}

const getEmotionColor = (e) => {
  const emotion = e?.toLowerCase() || '';
  if (['joy', 'love', 'happy'].some(x => emotion.includes(x))) return '#d8b4fe';
  if (['anger', 'disgust', 'hate'].some(x => emotion.includes(x))) return '#fca5a5';
  if (['sadness', 'fear', 'grief'].some(x => emotion.includes(x))) return '#7dd3fc';
  if (emotion.includes('surprise')) return '#67e8f9';
  return '#cbd5e1';
}

export default App
