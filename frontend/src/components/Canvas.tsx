import { useContext, useEffect, useRef, useState } from "react";
import AppContext, { Point } from "./interfaces";
import { colours } from "./interfaces";
import {
  replaceGreyscaleWithColours,
  getImagefromImageData,
} from "./imageLogic";
import { DrawStyle } from "./interfaces";

const ADDITIONAL_SF = 1;
const DEFAULT_ANGLE_RAD = 30 * (Math.PI / 180);

const TOP_FACE = "#f0f0f0";
const RIGHT_FACE = "#ababab64";
const LIGHT_GREY = "#838383d9";
const TRANS_RED = "#dc5e5e80";
const DARK_GREY = "#363636f0";

const rightSide: DrawStyle = {
  fillColour: RIGHT_FACE,
  lineColour: RIGHT_FACE,
  lineWidth: 4,
  toFill: true,
  lineCap: null,
  lineDash: null,
};
const rightSideFrame: DrawStyle = {
  fillColour: LIGHT_GREY,
  lineColour: LIGHT_GREY,
  lineWidth: 4,
  toFill: false,
  lineCap: "round",
  lineDash: [10, 10],
};
const topSide: DrawStyle = {
  fillColour: TOP_FACE,
  lineColour: TOP_FACE,
  lineWidth: 4,
  toFill: true,
  lineCap: null,
  lineDash: null,
};
const topSideFrame: DrawStyle = {
  fillColour: LIGHT_GREY,
  lineColour: LIGHT_GREY,
  lineWidth: 4,
  toFill: false,
  lineCap: "round",
  lineDash: [10, 10],
};

const centredStyle = {
  height: "75vh",
  width: "75vw", // was 60vh/w
  justifyContent: "center",
  alignItems: "center",
  padding: "10px",
  display: "flex",
  margin: "auto",
};

interface faces {
  face1: Array<Point>;
  face2: Array<Point>;
  face3: Array<Point>;
  dox: number;
  doy: number;
}

const get3DFacePoints = (
  ih: number,
  iw: number,
  id: number,
  theta: number,
  sfz: number,
  eps: number = 0,
): faces => {
  // get upper and right faces of a cube with front face of preview image and (projected) depth determined by theta and sfz
  const dox = Math.floor(Math.cos(theta) * id * sfz);
  const doy = Math.floor(Math.sin(theta) * id * sfz);
  const f1Points: Array<Point> = [
    { x: dox, y: eps },
    { x: iw + dox, y: eps },
    { x: iw, y: doy + eps },
    { x: 0, y: doy + eps },
  ];
  const f2Points: Array<Point> = [
    { x: iw + dox, y: eps },
    { x: iw, y: doy + eps },
    { x: iw, y: ih + doy - eps },
    { x: iw + dox, y: ih - eps },
  ];
  const f3Points: Array<Point> = [
    { x: 0, y: doy + eps },
    { x: iw, y: doy + eps },
    { x: iw, y: ih + doy - eps },
    { x: 0, y: ih + doy - eps },
  ];
  return {
    face1: f1Points,
    face2: f2Points,
    face3: f3Points,
    dox: dox,
    doy: doy,
  };
};

const getAspectCorrectedDims = (
  ih: number,
  iw: number,
  ch: number,
  cw: number,
  dox: number,
  doy: number,
  otherSF: number = 0.8,
) => {
  const hSF = ch / (ih + doy);
  const wSF = cw / (iw + dox);
  const maxFitSF = Math.min(hSF, wSF);
  const sf = maxFitSF * otherSF;
  const [nh, nw] = [ih * sf, iw * sf];
  return { w: nw, h: nh, ox: (cw - (nw + dox * sf)) / 2, oy: doy * sf, sf: sf };
};

const PreviewCanvas = () => {
  const {
    imageInfo: [imageInfo],
    previewImg: [previewImg, setPreviewImg],
    selectedPhase: [selectedPhase],
    targetL: [targetL],
    menuState: [menuState],
  } = useContext(AppContext)!;

  const containerRef = useRef<HTMLDivElement>(null);
  const frontDivRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);
  const [canvDims, setCanvDims] = useState<{ h: number; w: number }>({
    w: 300,
    h: 150,
  });

  const redraw = (image: HTMLImageElement | null) => {
    if (image === null) {
      return;
    } // null check - useful for the resize listeners
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d");
    if (ctx == null) {
      return;
    }
    ctx.imageSmoothingEnabled = false;
    const [ih, iw, ch, cw] = [
      image.naturalHeight,
      image.naturalWidth,
      canvas.height,
      canvas.width,
    ];
    const faceData = get3DFacePoints(
      ih,
      iw,
      imageInfo?.depth!,
      DEFAULT_ANGLE_RAD,
      0.3,
    );
    const correctDims = getAspectCorrectedDims(
      ih,
      iw,
      ch,
      cw,
      faceData.dox,
      faceData.doy,
      ADDITIONAL_SF,
    );
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
    if (faceData.dox > 0) {
      drawFaces(ctx!, faceData, correctDims.sf, correctDims.ox, correctDims.oy);
    }
    ctx?.drawImage(
      image,
      correctDims.ox,
      correctDims.oy,
      correctDims.w,
      correctDims.h,
    );
  };

  const drawFaces = (
    ctx: CanvasRenderingContext2D,
    faces: faces,
    sf: number,
    ox: number,
    oy: number,
    frame: boolean = false,
  ) => {
    const style1 = frame ? topSideFrame : topSide;
    const style2 = frame ? rightSideFrame : rightSide;
    drawPoints(ctx, faces.face1, sf, ox, oy, style1); //"#dbdbdbff"
    drawPoints(ctx, faces.face2, sf, ox, oy, style2);

    if (frame) {
      drawPoints(ctx, faces.face3, sf, ox, oy, style1);
    }
  };

  const drawPoints = (
    ctx: CanvasRenderingContext2D,
    points: Array<Point>,
    sf: number,
    ox: number,
    oy: number,
    style: DrawStyle,
  ) => {
    const p0 = points[0];
    ctx.fillStyle = style.fillColour;
    ctx.strokeStyle = style.lineColour;
    if (style.lineCap) {
      ctx.lineCap = style.lineCap;
    }
    if (style.lineDash) {
      ctx.setLineDash(style.lineDash);
    } else {
      ctx.setLineDash([]);
    }

    ctx.lineWidth = style.lineWidth;
    ctx.fillStyle = style.fillColour;
    ctx.beginPath();
    ctx.moveTo(ox + p0.x * sf, p0.y * sf);
    for (let i = 1; i < points.length; i++) {
      const p = points[i];
      ctx.lineTo(ox + p.x * sf, p.y * sf);
    }
    ctx.closePath();
    if (style.toFill) {
      ctx.fill();
    } else {
      ctx.stroke();
    }
  };

  const animateDiv = (div: HTMLDivElement, newW: number, newH: number) => {
    // Animate hidden bg div to show amount of data needed to measure
    div.style.width = `${newW}px`;
    div.style.height = `${newH}px`;
    div.style.outline = "10px #b4b4b4";
    div.style.borderRadius = "10px";
    div.style.backgroundColor = "#b4b4b4c2";
    //div.style.display = 'inline-block';
    // this creates the nice thick background lines
    div.style.background =
      "repeating-linear-gradient(45deg, #b4b4b4d9, #b4b4b4d9 10px, #e5e5e5e3 10px, #e5e5e5e3 20px)";

    div.animate(
      [
        // visiblity fade
        { opacity: 0 },
        { opacity: 1 },
      ],
      {
        fill: "both",
        duration: 2400,
        iterations: Infinity, // loop infinitely
        direction: "alternate", // forward and reverse
      },
    );
  };

  const animateCube = (
    hCanv: HTMLCanvasElement,
    newW: number,
    newH: number,
  ) => {
    hCanv.style.visibility = "visible";
    hCanv.width = newW;
    hCanv.height = newH;
    const hCanvCtx = hCanv.getContext("2d");
    const image = previewImg!;
    hCanvCtx?.clearRect(0, 0, newW, newH);
    const [ih, iw, ch, cw] = [
      image.naturalHeight,
      image.naturalWidth,
      hCanv.height,
      hCanv.width,
    ];
    const faceData = get3DFacePoints(
      ih,
      iw,
      imageInfo?.depth!,
      DEFAULT_ANGLE_RAD,
      0.3,
      1,
    );
    const correctDims = getAspectCorrectedDims(
      ih,
      iw,
      ch,
      cw,
      faceData.dox,
      faceData.doy,
      ADDITIONAL_SF,
    );
    if (faceData.dox > 0) {
      drawFaces(
        hCanvCtx!,
        faceData,
        correctDims.sf,
        correctDims.ox,
        correctDims.oy,
        true,
      );
    }

    hCanv.animate(
      [
        // visiblity fade
        { opacity: 0 },
        { opacity: 1 },
      ],
      {
        fill: "both",
        duration: 2400,
        iterations: Infinity, // loop infinitely
        direction: "alternate", // forward and reverse
      },
    );
  };

  // ================ EFFECTS ================
  useEffect(() => {
    // UPDATE WHEN IMAGE CHANGED
    redraw(previewImg!);
  }, [previewImg]);

  useEffect(() => {
    // HIGHLIGHT SPECIFIC PHASE WHEN BUTTON CLICKED
    if (imageInfo === null) {
      return;
    }
    const uniqueVals = imageInfo.phaseVals;
    // create mapping of {greyscaleVal1: [color to draw], greyscaleVal2: [color to draw]}
    // where color to draw is blue, orange, etc if val1 phase selected, greyscale otherwise
    const phaseCheck = (x: number, i: number) => {
      const alpha = selectedPhase == 0 ? 255 : 80;
      const originalColour = [x, [x, x, x, alpha]]; // identity mapping of grey -> grey in RGBA
      const newColour = [x, colours[i + 1]]; // grey -> phase colour
      const phaseIsSelected = i + 1 == selectedPhase;
      return phaseIsSelected ? newColour : originalColour;
    };
    // NB we ignore alpha in the draw call so doesn't matter that it's x here
    const mapping = Object.fromEntries(
      uniqueVals!.map((x, i, _) => phaseCheck(x, i)),
    );

    const imageData = imageInfo?.previewData;
    const newImageArr = replaceGreyscaleWithColours(imageData.data, mapping);
    const newImageData = new ImageData(
      newImageArr,
      imageInfo.width,
      imageInfo.height,
    );
    const newImage = getImagefromImageData(
      newImageData,
      imageInfo.height,
      imageInfo.width,
    );
    setPreviewImg(newImage);
  }, [selectedPhase]);

  useEffect(() => {
    // SET INITIAL CANV W AND H
    // runs on load to update canvDims with client rendered w and h of canvas (which is in vh/vw units)
    const resize = () => {
      const container = containerRef.current;
      const newCanvSize = {
        h: container!.clientHeight,
        w: container!.clientWidth,
      };
      setCanvDims(newCanvSize);
    };
    // will trigger whenever client resizes
    window.addEventListener("resize", resize);
    resize();
  }, []);

  useEffect(() => {
    // UPDATE WHEN CLIENT RESIZED
    const canv = canvasRef.current!;
    canv.width = canvDims.w;
    canv.height = canvDims.h;
    // we need to redraw otherwise setting w and h will clear the canvas
    redraw(previewImg);
  }, [canvDims]);

  useEffect(() => {
    // Animated shrink of canvas into top left corner of parent div
    // sf is original width / target length
    if (targetL === null) {
      return;
    }
    if (menuState != "conf_result") {
      return;
    }

    const canvas = canvasRef.current!;
    const image = previewImg!;
    const [ih, iw, ch, cw] = [
      image.naturalHeight,
      image.naturalWidth,
      canvDims.h,
      canvDims.w,
    ];
    const correctDims = getAspectCorrectedDims(
      ih,
      iw,
      ch,
      cw,
      0,
      0,
      ADDITIONAL_SF,
    );

    const shortestSideCanv = Math.min(canvDims.h, canvDims.w);
    const longestSideCanv = Math.max(correctDims.h, correctDims.w);
    const newCanvL = shortestSideCanv;

    const shortestSideData = Math.min(imageInfo?.width!, imageInfo?.height!);
    const longestSideData = Math.max(imageInfo?.width!, imageInfo?.height!);

    if (targetL < shortestSideData) {
      return;
    } // if already representative
    const newDataToCanv = newCanvL / targetL;

    const longestSideShrunkCanv = newDataToCanv * longestSideData;
    const maxSF = longestSideCanv / longestSideShrunkCanv;

    const canvAnim = canvas.animate(
      [{ transform: `scale(${1 / maxSF}) translate(${0}px, ${0}px)` }],
      {
        duration: 1600, //1.6s
        iterations: 1, // one shot
        fill: "forwards", // no reset
        easing: "ease-in-out",
      },
    );

    canvAnim.onfinish = (e) => {
      if (imageInfo?.nDims == 2) {
        animateDiv(frontDivRef.current!, newCanvL, newCanvL);
      } else {
        animateCube(hiddenCanvasRef.current!, canvDims.w, canvDims.h);
      }
    };
  }, [menuState]);
  return (
    <div ref={containerRef} style={centredStyle}>
      <div ref={frontDivRef} style={{ position: "absolute" }}></div>
      <canvas ref={canvasRef} id={"preview"}></canvas>
      <canvas
        ref={hiddenCanvasRef}
        style={{ visibility: "hidden", position: "absolute" }}
        id={"hidden"}
      ></canvas>
    </div>
  );
};

export default PreviewCanvas;
