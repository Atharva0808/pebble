"use client";

import { useEffect } from "react";

export default function Oneko() {
  useEffect(() => {
    // Avoid running on server or duplicate initializations
    if (typeof window === "undefined" || document.getElementById("oneko")) {
      return;
    }

    const isReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (isReducedMotion) return;

    const nekoEl = document.createElement("div");
    nekoEl.id = "oneko";

    let nekoPosX = 120;
    let nekoPosY = 120;
    let mousePosX = 300;
    let mousePosY = 300;
    let frameCount = 0;
    let idleTime = 0;
    let idleAnimation: string | null = null;
    let idleAnimationFrame = 0;
    const nekoSpeed = 10;

    const spriteSets: Record<string, number[][]> = {
      idle: [[-3, -3]],
      alert: [[-7, -3]],
      scratchSelf: [
        [-5, 0],
        [-6, 0],
        [-7, 0],
      ],
      scratchWallN: [
        [0, 0],
        [0, -1],
      ],
      scratchWallS: [
        [-4, -3],
        [-4, -2],
      ],
      scratchWallE: [
        [-2, -2],
        [-2, -3],
      ],
      scratchWallW: [
        [-4, -1],
        [-4, 0],
      ],
      tired: [[-3, -2]],
      sleeping: [
        [-2, 0],
        [-2, -1],
      ],
      N: [
        [-1, -2],
        [-1, -3],
      ],
      NE: [
        [0, -2],
        [0, -3],
      ],
      E: [
        [-3, 0],
        [-3, -1],
      ],
      SE: [
        [-5, -1],
        [-5, -2],
      ],
      S: [
        [-6, -3],
        [-7, -2],
      ],
      SW: [
        [-5, -3],
        [-6, -1],
      ],
      W: [
        [-4, -2],
        [-4, -3],
      ],
      NW: [
        [-1, 0],
        [-1, -1],
      ],
    };

    function setSprite(name: string, frame: number) {
      const sprite = spriteSets[name][frame % spriteSets[name].length];
      nekoEl.style.backgroundPosition = `${sprite[0] * 32}px ${sprite[1] * 32}px`;
    }

    function resetIdle() {
      idleAnimation = null;
      idleAnimationFrame = 0;
    }

    function idle() {
      idleTime += 1;

      if (
        idleTime > 10 &&
        Math.floor(Math.random() * 200) === 0 &&
        idleAnimation === null
      ) {
        const availableIdleAnimations = ["sleeping", "scratchSelf"];
        idleAnimation =
          availableIdleAnimations[
          Math.floor(Math.random() * availableIdleAnimations.length)
          ];
      }

      switch (idleAnimation) {
        case "sleeping":
          if (idleAnimationFrame < 8) {
            setSprite("tired", 0);
            break;
          }
          setSprite("sleeping", Math.floor(idleAnimationFrame / 8));
          if (idleAnimationFrame > 192) {
            resetIdle();
          }
          break;
        case "scratchSelf":
          setSprite("scratchSelf", Math.floor(idleAnimationFrame / 4));
          if (idleAnimationFrame > 36) {
            resetIdle();
          }
          break;
        default:
          setSprite("idle", 0);
          return;
      }
      idleAnimationFrame += 1;
    }

    function update() {
      frameCount += 1;
      const diffX = nekoPosX - mousePosX;
      const diffY = nekoPosY - mousePosY;
      const distance = Math.sqrt(diffX ** 2 + diffY ** 2);

      if (distance < nekoSpeed || distance < 24) {
        idle();
        return;
      }

      idleTime = 0;
      resetIdle();

      if (idleAnimationFrame > 0) {
        setSprite("alert", 0);
        idleAnimationFrame -= 1;
        return;
      }

      let direction = "";
      direction += diffY / distance > 0.5 ? "N" : "";
      direction += diffY / distance < -0.5 ? "S" : "";
      direction += diffX / distance > 0.5 ? "W" : "";
      direction += diffX / distance < -0.5 ? "E" : "";

      setSprite(direction || "idle", frameCount);

      nekoPosX -= (diffX / distance) * nekoSpeed;
      nekoPosY -= (diffY / distance) * nekoSpeed;

      nekoEl.style.left = `${nekoPosX - 16}px`;
      nekoEl.style.top = `${nekoPosY - 16}px`;
    }

    function onMouseMove(event: MouseEvent) {
      mousePosX = event.clientX;
      mousePosY = event.clientY;
    }

    nekoEl.style.width = "32px";
    nekoEl.style.height = "32px";
    nekoEl.style.position = "fixed";
    nekoEl.style.pointerEvents = "none";
    nekoEl.style.backgroundImage = "url('/oneko.gif')";
    nekoEl.style.imageRendering = "pixelated";
    nekoEl.style.zIndex = "999999";

    nekoEl.style.left = `${nekoPosX - 16}px`;
    nekoEl.style.top = `${nekoPosY - 16}px`;

    document.body.appendChild(nekoEl);

    document.addEventListener("mousemove", onMouseMove);
    const interval = setInterval(update, 66);

    return () => {
      document.removeEventListener("mousemove", onMouseMove);
      clearInterval(interval);
      nekoEl.remove();
    };
  }, []);

  return null;
}
