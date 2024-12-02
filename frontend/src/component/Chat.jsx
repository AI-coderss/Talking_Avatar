import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import useSpeechStore from "./store/useSpeechStore";
import ChatInputWidget from "./ChatInputWidget";
import "./Chat.css";

const Chat = () => {
  const [chats, setChats] = useState([
    { msg: "Hi there! How can I assist you today?", who: "bot" },
  ]);
  const [loading, setLoading] = useState(false);
  const [isChatVisible, setIsChatVisible] = useState(false);
  const chatContentRef = useRef(null);

  const setAudio = useSpeechStore((state) => state.setAudio);
  const setBlendshapes = useSpeechStore((state) => state.setBlendshapes);
  const { speak, setSpeak } = useSpeechStore();

  const scrollToBottom = () => {
    if (chatContentRef.current) {
      chatContentRef.current.scrollTo({
        top: chatContentRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [chats, loading]);

  const handleNewMessage = async (data) => {
    setIsChatVisible(true);

    if (data.text && data.text.trim().length > 0 && speak === false) {
      setChats((prevChats) => [...prevChats, { msg: data.text, who: "me" }]);
      setLoading(true);

      try {
        const response = await axios.post("http://127.0.0.1:5000/generate", {
          input: data.text,
        });

        const { text: botResponse, audio, blendshapes } = response.data;

        // Update chat with bot response
        setChats((prevChats) => [
          ...prevChats,
          { msg: botResponse, who: "bot" },
        ]);

        // Update Zustand store
        setAudio(audio);
        setBlendshapes(blendshapes);
        setSpeak(true); // Trigger Avatar speech
      } catch (error) {
        console.error("Error fetching response from /generate:", error);
        setChats((prevChats) => [
          ...prevChats,
          {
            msg: "Sorry, I couldn't process your request. Please try again.",
            who: "bot",
          },
        ]);
      } finally {
        setLoading(false);
      }
    }
  };

  const toggleChatVisibility = () => {
    setIsChatVisible((prevState) => !prevState);
  };

  return (
    <>
      {isChatVisible && (
        <div className="chat-content" ref={chatContentRef}>
          {chats.map((chat, index) => (
            <div key={index} className={`chat-message ${chat.who}`}>
              {chat.who === "bot" && (
                <figure className="avatar">
                  <img
                    src="https://i.ibb.co/bXnQgZg/BG-image.png"
                    alt="avatar"
                  />
                </figure>
              )}
              <div className="message-text">{chat.msg}</div>
            </div>
          ))}

          {loading && (
            <div className="chat-message loading">
              <figure className="avatar">
                <img
                  src="https://i.ibb.co/bXnQgZg/BG-image.png"
                  alt="avatar"
                />
              </figure>
              <div
                style={{ padding: "5px", display: "flex", alignItems: "center" }}
              >
                <lottie-player
                  src="https://lottie.host/d354a5c5-9a8b-456f-a7ed-e88fd09ce683/vYJTHMVdFJ.json"
                  style={{ width: "60px", height: "60px" }}
                  loop
                  autoplay
                  speed="1"
                  direction="1"
                  mode="normal"
                ></lottie-player>
              </div>
            </div>
          )}
        </div>
      )}
      <div className="chat-footer">
        <ChatInputWidget onSendMessage={handleNewMessage} />
        <button className="toggle-button" onClick={toggleChatVisibility}>
          {isChatVisible ? "-" : "+"}
        </button>
      </div>
    </>
  );
};

export default Chat;



