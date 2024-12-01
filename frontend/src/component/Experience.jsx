import React, { Suspense } from "react";
import { Environment, OrthographicCamera} from "@react-three/drei";

import { Canvas } from "@react-three/fiber";
import  Avatar  from "./Avatar";
import Bg from "./background";

export const Experience = () => {
  return (
    <>
     <Canvas>
        <OrthographicCamera makeDefault zoom={1400} position={[0, 1.65, 1]} />
        <Suspense fallback={null}>
          <Environment background={false} files="/images/photo_studio_loft_hall_1k.hdr" />
          <Bg />
          <Avatar avatar_url="/model.glb"/>
        </Suspense>
      </Canvas>
    </>
  );
};