import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, { imageLoadInfo } from "./interfaces";

const centredStyle = {
    height: '60vh', width: '60vw',
    justifyContent: 'center', alignItems: 'center',
    padding: '10px', display: 'flex', margin: 'auto',
}

const PreviewCanvas = () => {
    const {
        previewImg: [previewImg,],
    } = useContext(AppContext)!

    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [canvDims, setCanvDims] = useState<{ h: number, w: number }>({ w: 300, h: 150 });

    const redraw = (image: HTMLImageElement | null) => {
        if (image === null) { return; } // null check - useful for the resize listeners
        const canvas = canvasRef.current!;
        const ctx = canvas.getContext("2d");
        // TODO: ensure this works w/ image aspect ratio
        ctx?.drawImage(image, 0, 0, canvas.width, canvas.height);
    }

    // ================ EFFECTS ================
    useEffect(() => { // UPDATE WHEN IMAGE CHANGED
        redraw(previewImg!)
    }, [previewImg])

    useEffect(() => { // SET INITIAL CANV W AND H
        // runs on load to update canvDims with client rendered w and h of canvas (which is in vh/vw units)
        const resize = () => {
            const container = containerRef.current;
            const newCanvSize = { h: container!.clientHeight, w: container!.clientWidth };
            setCanvDims(newCanvSize);
        }
        // will trigger whenever client resizes
        window.addEventListener('resize', resize);
        resize();
    }, [])

    useEffect(() => { // UPDATE WHEN CLIENT RESIZED
        const canv = canvasRef.current!;
        canv.width = canvDims.w;
        canv.height = canvDims.h;
        // we need to redraw otherwise setting w and h will clear the canvas
        redraw(previewImg);
    }, [canvDims])

    return (
        <div ref={containerRef} style={centredStyle}>
            <canvas ref={canvasRef} id={"preview"}></canvas>
        </div>
    );
}

export default PreviewCanvas