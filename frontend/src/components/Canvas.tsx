import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext from "./interfaces";
import { colours } from "./interfaces";
import { replaceGreyscaleWithColours, getImagefromImageData } from "./imageLogic";

const centredStyle = {
    height: '60vh', width: '60vw',
    justifyContent: 'center', alignItems: 'center',
    padding: '10px', display: 'flex', margin: 'auto',
}

const getAspectCorrectedDims = (ih: number, iw: number, ch: number, cw: number, otherSF: number = 0.8) => {
    const hSF = ch / ih;
    const wSF = cw / iw;
    const sf = Math.min(hSF, wSF);
    const [nh, nw] = [ih * sf * otherSF, iw * sf * otherSF];
    return { w: nw, h: nh, ox: (cw - nw) / 2, oy: (ch - nh) / 2 };
}

const PreviewCanvas = () => {
    const {
        imageInfo: [imageInfo,],
        previewImg: [previewImg, setPreviewImg],
        selectedPhase: [selectedPhase,]
    } = useContext(AppContext)!

    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [canvDims, setCanvDims] = useState<{ h: number, w: number }>({ w: 300, h: 150 });

    const redraw = (image: HTMLImageElement | null) => {
        if (image === null) { return; } // null check - useful for the resize listeners
        const canvas = canvasRef.current!;
        const ctx = canvas.getContext("2d");
        const [ih, iw, ch, cw] = [image.naturalHeight, image.naturalWidth, canvas.height, canvas.width];
        const correctDims = getAspectCorrectedDims(ih, iw, ch, cw);
        ctx?.drawImage(image, correctDims.ox, correctDims.oy, correctDims.w, correctDims.h);
    }

    // ================ EFFECTS ================
    useEffect(() => { // UPDATE WHEN IMAGE CHANGED
        redraw(previewImg!)
    }, [previewImg])

    useEffect(() => {
        if (imageInfo === null) { return }
        const uniqueVals = imageInfo.phaseVals;
        const phaseCheck = (x: number, i: number) => {
            const originalColour = [x, [x, x, x, x]];
            const newColour = [x, colours[i + 1]];
            const phaseIsSelected = (i + 1 == selectedPhase);
            return phaseIsSelected ? newColour : originalColour;
        }
        const mapping = Object.fromEntries(
            uniqueVals!.map((x, i, _) => phaseCheck(x, i))
        );
        const imageData = imageInfo?.previewData;
        const newImageArr = replaceGreyscaleWithColours(imageData.data, mapping);
        const newImageData = new ImageData(newImageArr, imageInfo.width, imageInfo.height);
        const newImage = getImagefromImageData(newImageData, imageInfo.height, imageInfo.width);
        setPreviewImg(newImage);

        console.log(mapping)
    }, [selectedPhase])

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