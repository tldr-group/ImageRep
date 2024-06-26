import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext from "./interfaces";
import { colours } from "./interfaces";
import { dragDropStyle } from "./DragDrop"
import { replaceGreyscaleWithColours, getImagefromImageData } from "./imageLogic";

const ADDITIONAL_SF = 1

const centredStyle = {
    height: '75vh', width: '75vw', // was 60vh/w
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

const postZoomImageDims = (originalSize: number, targetL: number, cRect: DOMRect, pRect: DOMRect) => {
    //compute where corner of shrunk canv will be relative to pRect,
}

const PreviewCanvas = () => {
    const {
        imageInfo: [imageInfo,],
        previewImg: [previewImg, setPreviewImg],
        selectedPhase: [selectedPhase,],
        targetL: [targetL,]
    } = useContext(AppContext)!

    const containerRef = useRef<HTMLDivElement>(null);
    const animDivRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [canvDims, setCanvDims] = useState<{ h: number, w: number }>({ w: 300, h: 150 });

    const redraw = (image: HTMLImageElement | null) => {
        if (image === null) { return; } // null check - useful for the resize listeners
        const canvas = canvasRef.current!;
        const ctx = canvas.getContext("2d");
        const [ih, iw, ch, cw] = [image.naturalHeight, image.naturalWidth, canvas.height, canvas.width];
        const correctDims = getAspectCorrectedDims(ih, iw, ch, cw, ADDITIONAL_SF);
        ctx?.drawImage(image, correctDims.ox, correctDims.oy, correctDims.w, correctDims.h);
        //ctx?.drawImage(image, 0, 0, correctDims.w, correctDims.h);
    }

    const animateDiv = (newW: number, newH: number) => {
        // Animate hidden bg div to show amount of data needed to measure
        const div = animDivRef.current!;
        div.style.width = `${newW}px`
        div.style.height = `${newH}px`
        div.style.outline = '10px #b4b4b4';
        div.style.borderRadius = '10px';
        div.style.backgroundColor = '#b4b4b4c2';
        // this creates the nice thick background lines 
        div.style.background = "repeating-linear-gradient(45deg, #b4b4b4d9, #b4b4b4d9 10px, #e5e5e5e3 10px, #e5e5e5e3 20px)";

        div.animate([ // visiblity fade
            { opacity: 0 },
            { opacity: 1 },
        ], {
            fill: "both",
            duration: 2400,
            iterations: Infinity, // loop infinitely
            direction: "alternate" // forward and reverse
        })
    }

    // ================ EFFECTS ================
    useEffect(() => { // UPDATE WHEN IMAGE CHANGED
        redraw(previewImg!)
    }, [previewImg])

    useEffect(() => { // HIGHLIGHT SPECIFIC PHASE WHEN BUTTON CLICKED
        if (imageInfo === null) { return }
        const uniqueVals = imageInfo.phaseVals;
        // create mapping of {greyscaleVal1: [color to draw], greyscaleVal2: [color to draw]}
        // where color to draw is blue, orange, etc if val1 phase selected, greyscale otherwise
        const phaseCheck = (x: number, i: number) => {
            const originalColour = [x, [x, x, x, x]]; // identity mapping of grey -> grey in RGBA
            const newColour = [x, colours[i + 1]]; // grey -> phase colour
            const phaseIsSelected = (i + 1 == selectedPhase);
            return phaseIsSelected ? newColour : originalColour;
        }
        // NB we ignore alpha in the draw call so doesn't matter that it's x here
        const mapping = Object.fromEntries(
            uniqueVals!.map((x, i, _) => phaseCheck(x, i))
        );

        const imageData = imageInfo?.previewData;
        const newImageArr = replaceGreyscaleWithColours(imageData.data, mapping);
        const newImageData = new ImageData(newImageArr, imageInfo.width, imageInfo.height);
        const newImage = getImagefromImageData(newImageData, imageInfo.height, imageInfo.width);
        setPreviewImg(newImage);
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

    useEffect(() => {
        // Animated shrink of canvas into top left corner of parent div
        // sf is original width / target length
        if (targetL === null) { return; }
        const canvas = canvasRef.current!;
        const cRect = canvas.getBoundingClientRect();

        const sf = (targetL / imageInfo?.width!); // TODO: FIX WHEN DATA IS REAL

        const image = previewImg!
        const [ih, iw, ch, cw] = [image.naturalHeight, image.naturalWidth, canvas.height, canvas.width];
        const correctDims = getAspectCorrectedDims(ih, iw, ch, cw, ADDITIONAL_SF);
        // centred-adjusted shift
        const dx = (correctDims.w / 2) - (cRect.width / (2 * sf));
        const dy = (correctDims.h / 2) - (cRect.height / (2 * sf));

        // image drawn centred on canvas, need to correct to shift top left of image to top left of div.
        const shiftX = -(dx * sf + correctDims.ox);
        const shiftY = -(dy * sf + correctDims.oy);

        const canvAnim = canvas.animate([
            { transform: `scale(${1 / sf}) translate(${shiftX}px, ${shiftY}px)` },
        ], {
            duration: 1600, //1.6s
            iterations: 1, // one shot
            fill: 'forwards', // no reset 
            easing: 'ease-in-out'
        })
        canvAnim.onfinish = (e) => { animateDiv(correctDims.w, correctDims.h) }

    }, [targetL])

    return (
        <div ref={containerRef} style={centredStyle}>
            <div ref={animDivRef} style={{ position: 'absolute' }}></div>
            <canvas ref={canvasRef} id={"preview"}></canvas>
        </div>
    );
}

export default PreviewCanvas