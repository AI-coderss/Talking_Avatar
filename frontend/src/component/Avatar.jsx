// Avatar.jsx
import React, { useEffect, useMemo, useState } from 'react';
import { useGLTF, useTexture, useFBX, useAnimations } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { MeshStandardMaterial, MeshPhysicalMaterial, LineBasicMaterial, Vector2, LinearEncoding, sRGBEncoding } from 'three';
import * as THREE from 'three';
import makeSpeech from './makeSpeech';
import _, { set } from 'lodash';
import createAnimation from '../converter';
import blinkData from '../blendDataBlink.json';
import useSpeechStore from './store/useSpeechStore';

const host = 'http://127.0.0.1:5000';

export default function Avatar({ avatar_url }) {

  let gltf = useGLTF(avatar_url);
  let morphTargetDictionaryBody = null;
  let morphTargetDictionaryLowerTeeth = null;

  const [clips, setClips] = useState([]);
  const [audioBase64, setAudioBase64] = useState(null);

  const { audio, blendshapes, speak, setSpeak } = useSpeechStore();

  const [
    bodyTexture,
    eyesTexture,
    teethTexture,
    bodySpecularTexture,
    bodyRoughnessTexture,
    bodyNormalTexture,
    teethNormalTexture,
    // teethSpecularTexture,
    hairTexture,
    tshirtDiffuseTexture,
    tshirtNormalTexture,
    tshirtRoughnessTexture,
    hairAlphaTexture,
    hairNormalTexture,
    hairRoughnessTexture,
  ] = useTexture([
    "/images/body.webp",
    "/images/eyes.webp",
    "/images/teeth_diffuse.webp",
    "/images/body_specular.webp",
    "/images/body_roughness.webp",
    "/images/body_normal.webp",
    "/images/teeth_normal.webp",
    // "/images/teeth_specular.webp",
    "/images/h_color.webp",
    "/images/tshirt_diffuse.webp",
    "/images/tshirt_normal.webp",
    "/images/tshirt_roughness.webp",
    "/images/h_alpha.webp",
    "/images/h_normal.webp",
    "/images/h_roughness.webp",
  ]);

  _.each([
    bodyTexture,
    eyesTexture,
    teethTexture,
    teethNormalTexture,
    bodySpecularTexture,
    bodyRoughnessTexture,
    bodyNormalTexture,
    tshirtDiffuseTexture,
    tshirtNormalTexture,
    tshirtRoughnessTexture,
    hairAlphaTexture,
    hairNormalTexture,
    hairRoughnessTexture
  ], t => {
    t.encoding = sRGBEncoding;
    t.flipY = false;
  });

  bodyNormalTexture.encoding = LinearEncoding;
  tshirtNormalTexture.encoding = LinearEncoding;
  teethNormalTexture.encoding = LinearEncoding;
  hairNormalTexture.encoding = LinearEncoding;


  gltf.scene.traverse(node => {
    if (node.type === 'Mesh' || node.type === 'LineSegments' || node.type === 'SkinnedMesh') {
      node.castShadow = true;
      node.receiveShadow = true;
      node.frustumCulled = false;
      if (node.name.includes("Body")) {

        node.castShadow = true;
        node.receiveShadow = true;

        node.material = new MeshPhysicalMaterial();
        node.material.map = bodyTexture;
        // node.material.shininess = 60;
        node.material.roughness = 1.7;

        // node.material.specularMap = bodySpecularTexture;
        node.material.roughnessMap = bodyRoughnessTexture;
        node.material.normalMap = bodyNormalTexture;
        node.material.normalScale = new Vector2(0.6, 0.6);

        morphTargetDictionaryBody = node.morphTargetDictionary;

        node.material.envMapIntensity = 0.8;
        // node.material.visible = false;

      }

      if (node.name.includes("Eyes")) {
        node.material = new MeshStandardMaterial();
        node.material.map = eyesTexture;
        // node.material.shininess = 100;
        node.material.roughness = 0.1;
        node.material.envMapIntensity = 0.5;


      }

      if (node.name.includes("Brows")) {
        node.material = new LineBasicMaterial({ color: 0x000000 });
        node.material.linewidth = 1;
        node.material.opacity = 0.5;
        node.material.transparent = true;
        node.visible = false;
      }

      if (node.name.includes("Teeth")) {

        node.receiveShadow = true;
        node.castShadow = true;
        node.material = new MeshStandardMaterial();
        node.material.roughness = 0.1;
        node.material.map = teethTexture;
        node.material.normalMap = teethNormalTexture;

        node.material.envMapIntensity = 0.7;


      }

      if (node.name.includes("Hair")) {
        node.material = new MeshStandardMaterial();
        node.material.map = hairTexture;
        node.material.alphaMap = hairAlphaTexture;
        node.material.normalMap = hairNormalTexture;
        node.material.roughnessMap = hairRoughnessTexture;

        node.material.transparent = true;
        node.material.depthWrite = false;
        node.material.side = 2;
        node.material.color.setHex(0x000000);

        node.material.envMapIntensity = 0.3;


      }

      if (node.name.includes("TSHIRT")) {
        node.material = new MeshStandardMaterial();

        node.material.map = tshirtDiffuseTexture;
        node.material.roughnessMap = tshirtRoughnessTexture;
        node.material.normalMap = tshirtNormalTexture;
        node.material.color.setHex(0xffffff);

        node.material.envMapIntensity = 0.5;


      }

      if (node.name.includes("TeethLower")) {
        morphTargetDictionaryLowerTeeth = node.morphTargetDictionary;
      }

    }

  });

  const playBase64Audio = (base64Audio) => {
    try {
      // Ensure the string has the correct base64 prefix
      if (!base64Audio.startsWith('data:')) {
        base64Audio = `data:audio/mp3;base64,${base64Audio}`;
      }

      // Split and decode the base64 string
      const base64Data = base64Audio.split(',')[1]; // Remove the prefix
      if (!base64Data) {
        throw new Error('Base64 string is missing data after the prefix.');
      }

      const byteString = atob(base64Data); // Decode the base64 string
      const arrayBuffer = new Uint8Array(byteString.length);

      for (let i = 0; i < byteString.length; i++) {
        arrayBuffer[i] = byteString.charCodeAt(i);
      }

      // Create a Blob and a URL for it
      const audioBlob = new Blob([arrayBuffer], { type: 'audio/mp3' }); // Use correct MIME type
      const audioUrl = URL.createObjectURL(audioBlob);

      // Create an Audio element
      const audioElement = new Audio(audioUrl);

      // Use canplaythrough to wait until the audio is fully ready to play
      audioElement.addEventListener('canplaythrough', () => {
        console.log('Audio is ready to play.');
        setSpeak(true); // Set speak to true

        setTimeout(() => {
          let newClips = [
            createAnimation(blendshapes, morphTargetDictionaryBody, 'HG_Body'),
            createAnimation(blendshapes, morphTargetDictionaryLowerTeeth, 'HG_TeethLower'),
          ];

          setClips(newClips);
        }, 350);

        audioElement.play().catch((err) => {
          console.error('Error playing audio:', err);
        });
      });

      // Set speak to false after audio ends
      audioElement.addEventListener('ended', () => {
        setSpeak(false); // Set speak to false
      });

      // Load the audio
      audioElement.load();
    } catch (err) {
      console.error('Failed to play audio:', err);
    }
  };

  useEffect(() => {
    if (audio) {
      // Play base64 audio
      playBase64Audio(audio);
    }
  }, [audioBase64]);

  const mixer = useMemo(() => new THREE.AnimationMixer(gltf.scene), []);

  let idleFbx = useFBX('/idle.fbx');
  let { clips: idleClips } = useAnimations(idleFbx.animations);

  idleClips[0].tracks = _.filter(idleClips[0].tracks, track => {
    return track.name.includes("Head") || track.name.includes("Neck") || track.name.includes("Spine2");
  });

  idleClips[0].tracks = _.map(idleClips[0].tracks, track => {

    if (track.name.includes("Head")) {
      track.name = "head.quaternion";
    }

    if (track.name.includes("Neck")) {
      track.name = "neck.quaternion";
    }

    if (track.name.includes("Spine")) {
      track.name = "spine2.quaternion";
    }

    return track;

  });

  useEffect(() => {

    let idleClipAction = mixer.clipAction(idleClips[0]);
    idleClipAction.play();

    let blinkClip = createAnimation(blinkData, morphTargetDictionaryBody, 'HG_Body');
    let blinkAction = mixer.clipAction(blinkClip);
    blinkAction.play();
  }, []);

  // Play animation clips when available
  useEffect(() => {
    if (speak === false)
      return;

    _.each(clips, clip => {
      let clipAction = mixer.clipAction(clip);
      clipAction.setLoop(THREE.LoopOnce);
      clipAction.play();

    });

  }, [clips, mixer, audio]);

  useFrame((state, delta) => {
    mixer.update(delta);
  });

  useEffect(() => {
    console.log(speak);
    try {
      if (speak === false)
        return;
      setAudioBase64(audio);
    } catch (err) {
      console.error(err);
    }
  }, [audio]);

  return (
    <group name="avatar">
      <primitive object={gltf.scene} dispose={null} />
    </group>
  );
}
