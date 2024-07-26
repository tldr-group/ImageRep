import { useEffect, useRef, useState } from "react";

const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 350;
const H_FOR_TEXT = 20;
const SCALEBAR_WIDTH = 4;
const H_GAUSS = CANVAS_HEIGHT - H_FOR_TEXT


const xPxToData = (idx: number, lb_data: number, ub_data: number, canv_w: number) => {
    return (idx / (canv_w)) * (ub_data - lb_data) + lb_data
};

const xDataToPx = (x: number, lb_data: number, ub_data: number, canv_w: number) => {
    return Math.round(((x - lb_data) / (ub_data - lb_data)) * canv_w)
};

const yDataToPx = (y: number, lb_data: number, ub_data: number, canv_h: number) => {
    return canv_h - ((y / (ub_data - lb_data)) * canv_h)
};

const normalDist = (x: number, mu: number, sigma: number) => {
    return (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow(((x - mu) / sigma), 2))
};

const indices = [...Array(CANVAS_WIDTH).keys()];

const tmpMu = 0.4;
const tmpSigma = 0.08;
const [tmpStart, tmpEnd] = [0.2, 0.6];
const tmpMax = normalDist(tmpMu, tmpMu, tmpSigma);


interface DrawStyle {
    fillColour: string,
    lineColour: string,
    lineWidth: number,
    toFill: boolean,
    lineCap: CanvasPathDrawingStyles["lineCap"] | null,
    lineDash: Array<number> | null,
}

interface NormalParams {
    mu: number,
    sigma: number,
    start_pf: number,
    end_pf: number,
    max_y: number
}

const LIGHT_GREY = "#838383d9"
const TRANS_RED = "#dc5e5e80"

const xAxisStyle: DrawStyle = { fillColour: 'black', lineColour: LIGHT_GREY, lineWidth: 3, toFill: false, lineCap: null, lineDash: null }
const yAxisStyle: DrawStyle = { fillColour: 'black', lineColour: LIGHT_GREY, lineWidth: 3, toFill: false, lineCap: null, lineDash: [4, 10] }
const curveStyle: DrawStyle = { fillColour: 'red', lineColour: 'red', lineWidth: 3, toFill: false, lineCap: null, lineDash: null }
const shadeStyle: DrawStyle = { fillColour: TRANS_RED, lineColour: TRANS_RED, lineWidth: 3, toFill: true, lineCap: null, lineDash: null }

const NormalSlider = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [params, setParams] = useState<NormalParams>({ mu: tmpMu, sigma: tmpSigma, start_pf: tmpStart, end_pf: tmpEnd, max_y: tmpMax });

    const getShadedPoints = (newLB: number, newUB: number) => {
        const pxLBx = xDataToPx(newLB, params.start_pf, params.end_pf, CANVAS_WIDTH);
        const pxUBx = xDataToPx(newUB, params.start_pf, params.end_pf, CANVAS_WIDTH);

        const nNew = pxUBx - pxLBx;
        const inds = [...Array(nNew).keys()].map((x) => (x + pxLBx));
        const xData = inds.map((x) => xPxToData((x), params.start_pf, params.end_pf, CANVAS_WIDTH));
        const xPadded = [inds[0]].concat(inds, [inds[nNew - 1]])

        const yData = xData.map((xd) => normalDist(xd, params.mu, params.sigma));
        const maxY = normalDist(params.mu, params.mu, params.sigma);
        const yPoints = yData.map((yd) => yDataToPx(yd, 0, maxY + 0.2, H_GAUSS));
        const yPadded = [H_GAUSS].concat(yPoints, [H_GAUSS]);
        return { xPoints: xPadded, yPoints: yPadded };
    }

    const drawPoints = (xPoints: Array<number>, yPoints: Array<number>, style: DrawStyle) => {
        const canv = canvasRef.current!;
        const ctx = canv.getContext('2d')!;

        ctx.fillStyle = style.fillColour;
        ctx.strokeStyle = style.lineColour;

        if (style.lineCap) { ctx.lineCap = style.lineCap; }
        if (style.lineDash) {
            ctx.setLineDash(style.lineDash)
        } else {
            ctx.setLineDash([])
        }

        ctx.lineWidth = style.lineWidth;
        ctx.beginPath();
        ctx.moveTo(xPoints[0], yPoints[0]);
        for (let i = 0; i < xPoints.length; i++) {
            const x = xPoints[i];
            const y = yPoints[i];
            ctx.lineTo(x, y);
        }
        if (style.toFill) {
            ctx.closePath();
            ctx.fill();
        } else {
            ctx.stroke();
        }
    }

    const drawText = (mu: number,) => { }

    useEffect(() => {
        // generate, draw
        const xData = indices.map((x) => xPxToData((x), params.start_pf, params.end_pf, CANVAS_WIDTH));
        const yData = xData.map((xd) => normalDist(xd, params.mu, params.sigma));
        const maxY = normalDist(params.mu, params.mu, params.sigma);

        const yPoints = yData.map((yd) => yDataToPx(yd, 0, maxY + 0.2, H_GAUSS));
        const sp = getShadedPoints(0.35, 0.45);
        console.log(sp)

        drawPoints([0, CANVAS_WIDTH], [H_GAUSS, H_GAUSS], xAxisStyle)
        drawPoints([CANVAS_WIDTH / 2, CANVAS_WIDTH / 2], [0, H_GAUSS], yAxisStyle)
        drawPoints(indices, yPoints, curveStyle);
        drawPoints(sp.xPoints, sp.yPoints, shadeStyle);
    }, [])

    return (
        <div>
            <canvas width={CANVAS_WIDTH} height={CANVAS_HEIGHT} ref={canvasRef} />
        </div>)
}

export default NormalSlider