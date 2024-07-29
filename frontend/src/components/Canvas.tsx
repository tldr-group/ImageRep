import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, { Point } from "./interfaces";
import { colours } from "./interfaces";
import { replaceGreyscaleWithColours, getImagefromImageData } from "./imageLogic";

const ADDITIONAL_SF = 1
const DEFAULT_ANGLE_RAD = 30 * (Math.PI / 180)

const centredStyle = {
    height: '75vh', width: '75vw', // was 60vh/w
    justifyContent: 'center', alignItems: 'center',
    padding: '10px', display: 'flex', margin: 'auto',
}

const get3DFacePoints = (ih: number, iw: number, id: number, theta: number, sfz: number) => {
    // get upper and right faces of a cube with front face of preview image and (projected) depth determined by theta and sfz 
    const dox = Math.floor((Math.cos(theta) * id) * sfz);
    const doy = Math.floor((Math.sin(theta) * id) * sfz);
    const f1Points: Array<Point> = [{ x: dox, y: 0 }, { x: iw + dox, y: 0 }, { x: iw, y: doy }, { x: 0, y: doy }];
    const f2Points: Array<Point> = [{ x: iw + dox, y: 0 }, { x: iw, y: doy }, { x: iw, y: ih + doy }, { x: iw + dox, y: ih }];
    return { face1: f1Points, face2: f2Points, dox: dox, doy: doy };
}

const getAspectCorrectedDims = (ih: number, iw: number, ch: number, cw: number, dox: number, doy: number, otherSF: number = 0.8) => {
    const hSF = (ch) / (ih + doy);
    const wSF = (cw) / (iw + dox);
    const maxFitSF = Math.min(hSF, wSF);
    const sf = maxFitSF * otherSF
    const [nh, nw] = [ih * sf, iw * sf];
    return { w: nw, h: nh, ox: ((cw - (nw + dox * sf)) / 2), oy: doy * sf, sf: sf };
}


const PreviewCanvas = () => {
    const {
        imageInfo: [imageInfo,],
        previewImg: [previewImg, setPreviewImg],
        selectedPhase: [selectedPhase,],
        targetL: [targetL,],
        menuState: [menuState,]
    } = useContext(AppContext)!

    const containerRef = useRef<HTMLDivElement>(null);
    const frontDivRef = useRef<HTMLDivElement>(null);
    const topDivRef = useRef<HTMLDivElement>(null);
    const sideDivRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [canvDims, setCanvDims] = useState<{ h: number, w: number }>({ w: 300, h: 150 });

    const redraw = (image: HTMLImageElement | null) => {
        if (image === null) { return; } // null check - useful for the resize listeners
        const canvas = canvasRef.current!;
        const ctx = canvas.getContext("2d");
        const [ih, iw, ch, cw] = [image.naturalHeight, image.naturalWidth, canvas.height, canvas.width];
        const faceData = get3DFacePoints(ih, iw, imageInfo?.depth!, DEFAULT_ANGLE_RAD, 0.3);
        const correctDims = getAspectCorrectedDims(ih, iw, ch, cw, faceData.dox, faceData.doy, ADDITIONAL_SF);
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
        if (faceData.dox > 0) {
            console.log(faceData)
            console.log(correctDims.oy)
            drawFaces(ctx!, faceData.face1, faceData.face2, correctDims.sf, correctDims.ox, correctDims.oy)
        }
        ctx?.drawImage(image, correctDims.ox, correctDims.oy, correctDims.w, correctDims.h);
        //ctx?.drawImage(image, 0, 0, correctDims.w, correctDims.h);
    }

    const drawFaces = (ctx: CanvasRenderingContext2D, face1: Array<Point>, face2: Array<Point>, sf: number, ox: number, oy: number) => {
        drawPoints(ctx, face1, sf, ox, oy, "#f0f0f0"); //"#dbdbdbff" 
        drawPoints(ctx, face2, sf, ox, oy, "#ababab64");
    }

    const drawPoints = (ctx: CanvasRenderingContext2D, points: Array<Point>, sf: number, ox: number, oy: number, fill: string) => {
        const p0 = points[0];
        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.moveTo(ox + p0.x * sf, p0.y * sf);
        for (let i = 1; i < points.length; i++) {
            const p = points[i];
            ctx.lineTo(ox + p.x * sf, p.y * sf);
        }
        ctx.closePath();
        ctx.fill();
    }

    const animateDiv = (div: HTMLDivElement, newW: number, newH: number) => {
        // Animate hidden bg div to show amount of data needed to measure
        div.style.width = `${newW}px`
        div.style.height = `${newH}px`
        div.style.outline = '10px #b4b4b4';
        div.style.borderRadius = '10px';
        div.style.backgroundColor = '#b4b4b4c2';
        //div.style.display = 'inline-block';
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
            const alpha = (selectedPhase == 0) ? 255 : 80
            const originalColour = [x, [x, x, x, alpha]]; // identity mapping of grey -> grey in RGBA
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
        if (menuState != 'conf_result') { return }
        const canvas = canvasRef.current!;

        const shortestSide = Math.min(imageInfo?.width!, imageInfo?.height!);
        if (targetL < shortestSide) { return }; // if already representative
        const maxSF = (targetL / shortestSide);

        const newCanvL = Math.min(canvDims.h, canvDims.w)

        const image = previewImg!
        const [ih, iw, ch, cw] = [image.naturalHeight, image.naturalWidth, canvDims.h, canvDims.w];
        const correctDims = getAspectCorrectedDims(ih, iw, ch, cw, 0, 0, ADDITIONAL_SF);
        // centred-adjusted shift
        const dx = (correctDims.w / 2) - (newCanvL / (2 * maxSF));
        const dy = (correctDims.h / 2) - (newCanvL / (2 * maxSF));
        // image drawn centred on canvas, need to correct to shift top left of image to top left of div.
        const shiftX = -(dx * maxSF + correctDims.ox);
        const shiftY = -(dy * maxSF + correctDims.oy);


        console.log(correctDims, dx, dy)
        const canvAnim = canvas.animate([
            { transform: `scale(${1 / maxSF}) translate(${0}px, ${0}px)` },
        ], {
            duration: 1600, //1.6s
            iterations: 1, // one shot
            fill: 'forwards', // no reset 
            easing: 'ease-in-out'
        })
        canvAnim.onfinish = (e) => {
            animateDiv(frontDivRef.current!, newCanvL, newCanvL)
        }

    }, [menuState])

    return (
        <div ref={containerRef} style={centredStyle}>
            <div ref={frontDivRef} style={{ position: 'absolute' }}></div>
            <div ref={topDivRef} style={{ position: 'absolute' }}></div>
            <div ref={sideDivRef} style={{ position: 'absolute' }}></div>
            <canvas ref={canvasRef} id={"preview"}></canvas>
        </div>
    );
}

export default PreviewCanvas