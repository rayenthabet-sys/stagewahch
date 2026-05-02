import React, { useState, useEffect } from 'react';
import VisionSidebar from './components/VisionSidebar';
import ChatSection from './components/ChatSection';

const TM_URL = "https://teachablemachine.withgoogle.com/models/OxrRY4_kR/";

export default function App() {
  const [model, setModel] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [messages, setMessages] = useState([
    { role: 'ai', content: 'عسلامة خويا الفلاح! كيفاش نجم نعاونك اليوم؟ أبعثلي تسجيل صوتي وتصويرة كان لزم.' }
  ]);
  
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [selectedAudio, setSelectedAudio] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Load Teachable Machine
  useEffect(() => {
    async function loadModel() {
      try {
        const modelURL = TM_URL + "model.json";
        const metadataURL = TM_URL + "metadata.json";
        if (window.tmImage) {
          const loadedModel = await window.tmImage.load(modelURL, metadataURL);
          setModel(loadedModel);
          setIsModelLoading(false);
        }
      } catch (err) {
        console.error("Failed to load TM model", err);
      }
    }
    loadModel();
  }, []);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      const url = URL.createObjectURL(file);
      setSelectedImage(url);
      setPrediction(null);
      
      const img = new window.Image();
      img.src = url;
      img.onload = async () => {
        if (model) {
          const preds = await model.predict(img);
          setPrediction(preds.sort((a, b) => b.probability - a.probability));
        }
      };
    }
  };

  const handleAudioChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setAudioFile(file);
      setSelectedAudio(URL.createObjectURL(file));
    }
  };

  const removeImage = () => {
    setSelectedImage(null);
    setImageFile(null);
    setPrediction(null);
  };

  const removeAudio = () => {
    setSelectedAudio(null);
    setAudioFile(null);
  };

  const handleSend = async () => {
    if (!audioFile) {
      alert("Please upload a voice recording (.wav, .mp3, etc) for the AI!");
      return;
    }

    const userMessage = { 
      role: 'user', 
      content: 'Uploaded audio query.', 
      image: selectedImage,
      audioBaseUrl: selectedAudio
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsProcessing(true);

    try {
      const formData = new FormData();
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      let endpoint = `${API_URL}/api/process-audio`;
      
      if (imageFile) {
        endpoint = `${API_URL}/api/process-image`;
        formData.append('voice_query', audioFile);
        formData.append('images', imageFile);
      } else {
        formData.append('file', audioFile);
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const aiResponse = { 
        role: 'ai', 
        content: data.response_darija || "No response text found.",
        audioBase64: data.audio_base64
      };
      
      setMessages(prev => [...prev, aiResponse]);
      removeImage();
      removeAudio();
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'ai', content: 'Sama7ni, we encountered an error connecting to the backend. Make sure uvicorn is running!' }]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="main-layout">
      <VisionSidebar 
        isModelLoading={isModelLoading} 
        prediction={prediction} 
      />
      <ChatSection 
        messages={messages}
        isProcessing={isProcessing}
        selectedImage={selectedImage}
        selectedAudio={selectedAudio}
        audioFile={audioFile}
        handleSend={handleSend}
        handleImageChange={handleImageChange}
        handleAudioChange={handleAudioChange}
        removeImage={removeImage}
        removeAudio={removeAudio}
      />
    </div>
  );
}
