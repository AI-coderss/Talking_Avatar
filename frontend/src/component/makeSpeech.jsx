// import axios from "axios";
// import useSpeechStore from "./store/useSpeechStore";


// const makeSpeech = async (text) => {
//   const setAudio = useSpeechStore.getState().setAudio;
//   const setBlendshapes = useSpeechStore.getState().setBlendshapes;
//   const setSpeak = useSpeechStore.getState().setSpeak;


//   try {
//     const response = await axios.post("http://127.0.0.1:5000/generate", {
//       input: text,
//     });

//     const { audio, blendshapes } = response.data;

//     // Update Zustand store
//     setAudio(audio);
//     setBlendshapes(blendshapes);
//     setSpeak(true); // Trigger the Avatar to speak
//   } catch (error) {
//     console.error("Error in makeSpeech:", error);
//   }
// };

// export default makeSpeech;
