import { useEffect, useRef, useState } from "react";

const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 350;
const H_FOR_TEXT = 20;
const SCALEBAR_WIDTH = 4;
const H_GAUSS = CANVAS_HEIGHT - H_FOR_TEXT


const xPxToData = (idx: number, lb_data: number, ub_data: number, canv_w: number) => {
    return (idx / canv_w) * (ub_data - lb_data) + lb_data
};

const yDataToPx = (y: number, lb_data: number, ub_data: number, canv_h: number) => {
    return canv_h - ((y / (ub_data - lb_data)) * canv_h)
};

const normalDist = (x: number, mu: number, sigma: number) => {
    return (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow(((x - mu) / sigma), 2))
};

const indices = [...Array(CANVAS_WIDTH).keys()];


const NormalSlider = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const drawPoints = (xPoints: Array<number>, yPoints: Array<number>, lw: number, colour: string, fill: boolean = false) => {
        const canv = canvasRef.current!;
        const ctx = canv.getContext('2d')!;

        ctx.fillStyle = colour;
        ctx.strokeStyle = colour;
        ctx.lineWidth = lw;
        ctx.beginPath();
        ctx.moveTo(xPoints[0], yPoints[0]);
        for (let i = 0; i < xPoints.length; i++) {
            const x = xPoints[i];
            const y = yPoints[i];
            ctx.lineTo(x, y);
        }
        if (fill) {
            ctx.closePath();
            ctx.fill();
        } else {
            ctx.stroke();
        }
    }

    useEffect(() => {
        // these are application dependent
        const start_pf = 0.2;
        const end_pf = 0.6;
        const mu = 0.4;
        const sigma = 0.1;
        // generate, draw
        const xData = indices.map((x) => xPxToData((x), start_pf, end_pf, CANVAS_WIDTH));
        const yData = xData.map((xd) => normalDist(xd, mu, sigma));
        const maxY = normalDist(mu, mu, sigma);

        const yPoints = yData.map((yd) => yDataToPx(yd, 0, maxY + 0.2, H_GAUSS));
        drawPoints(indices, yPoints, 3, 'red',)
    }, [])

    return (
        <div>
            <canvas width={CANVAS_WIDTH} height={CANVAS_HEIGHT} ref={canvasRef} />
        </div>)
}

export default NormalSlider