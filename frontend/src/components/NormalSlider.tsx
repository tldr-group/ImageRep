import { useEffect, useRef, useState, useContext } from "react";
import AppContext, { rgbaToHex, colours } from "./interfaces";
import { getPhaseFraction } from "./imageLogic";

import InputGroup from "react-bootstrap/InputGroup";
import Form from "react-bootstrap/Form";
import { head } from "underscore";

const CANVAS_WIDTH = 428 * 2;
const CANVAS_HEIGHT = 240 * 1.5;
const H_FOR_TEXT = 80;
const TEXT_OFFSET = 30;
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

const getPredictionInterval = (conf: number, pf1D: Array<number>, cumSumSum: Array<number>) => {
    const halfConfLevel = (1 + conf) / 2;

    const ltConf = cumSumSum.map(x => x > ((1 - halfConfLevel)));
    const gtConf = cumSumSum.map(x => x > halfConfLevel);

    const ltConfIdx = ltConf.findIndex((x) => x)
    const gtConfIdx = gtConf.findIndex((x) => x)
    return [pf1D[ltConfIdx], pf1D[gtConfIdx]]
}

const indices = [...Array(CANVAS_WIDTH).keys()];

const tmpMu = 0.4;
const tmpSigma = 0.08;
const [tmpStart, tmpEnd] = [0.2, 0.6];
const tmpMax = normalDist(tmpMu, tmpMu, tmpSigma);
const [tmpLB, tmpUB] = [tmpMu - 0.05, tmpMu + 0.05]


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
    max_y: number,
}

const LIGHT_GREY = "#838383d9"
const TRANS_RED = "#dc5e5e80"
const DARK_GREY = "#363636f0"

const xAxisStyle: DrawStyle = { fillColour: LIGHT_GREY, lineColour: LIGHT_GREY, lineWidth: 3, toFill: false, lineCap: null, lineDash: null }
const yAxisStyle: DrawStyle = { fillColour: LIGHT_GREY, lineColour: LIGHT_GREY, lineWidth: 3, toFill: false, lineCap: null, lineDash: [4, 10] }
const curveStyle: DrawStyle = { fillColour: 'red', lineColour: 'red', lineWidth: 4, toFill: false, lineCap: null, lineDash: null }
const shadeStyle: DrawStyle = { fillColour: TRANS_RED, lineColour: TRANS_RED, lineWidth: 3, toFill: true, lineCap: null, lineDash: null }

const NormalSlider = () => {
    const {
        imageInfo: [imageInfo,],
        analysisInfo: [analysisInfo,],
        selectedPhase: [selectedPhase,],
        selectedConf: [selectedConf, setSelectedConf],
        errVF: [errVF, setErrVF],
        pfB: [pfB, setPfB],
        accurateFractions: [accurateFractions,],
    } = useContext(AppContext)!

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [params, setParams] = useState<NormalParams>({
        mu: tmpMu,
        sigma: tmpSigma,
        start_pf: tmpStart,
        end_pf: tmpEnd,
        max_y: tmpMax,
    });

    const c = colours[selectedPhase];
    const headerHex = rgbaToHex(c[0], c[1], c[2], c[3]);
    const shadedHex = rgbaToHex(c[0], c[1], c[2], 120);

    const vals = imageInfo?.phaseVals!
    const phaseFrac = (accurateFractions != null) ?
        accurateFractions[vals[selectedPhase - 1]]
        : getPhaseFraction(
            imageInfo?.previewData.data!,
            vals[selectedPhase - 1]
        );


    const getShadedPoints = (newLB: number, newUB: number) => {
        const pxLBx = xDataToPx(newLB, params.start_pf, params.end_pf, CANVAS_WIDTH);
        const pxUBx = xDataToPx(newUB, params.start_pf, params.end_pf, CANVAS_WIDTH);

        console.log(pxLBx, pxUBx, newLB, newUB)

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

    const clearCanv = () => {
        const canv = canvasRef.current!;
        const ctx = canv.getContext('2d')!;
        ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
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

    const drawLabels = () => {
        const canv = canvasRef.current!;
        const ctx = canv.getContext('2d')!;
        ctx.font = "28px Noto Sans"; // was 24
        ctx.fillStyle = headerHex;
        ctx.fillText("Likelihood of the", 10, 40);
        ctx.fillText("material's p.f.", 24, 70);

        const epsY = 24
        ctx.fillStyle = DARK_GREY;
        ctx.fillText('p.f. of image', CANVAS_WIDTH / 2 - 60, CANVAS_HEIGHT - H_FOR_TEXT / 2 + epsY);
        ctx.fillText('Phase Fraction (p.f.)', 10, CANVAS_HEIGHT - H_FOR_TEXT / 2 + epsY);
    }

    const drawText = (dataVals: Array<number>, xPositions: Array<number>, yOffset: number = TEXT_OFFSET, fontSize: number = 32) => {
        const canv = canvasRef.current!;
        const ctx = canv.getContext('2d')!;
        ctx.font = `${fontSize}px Noto Sans`; // was 24
        ctx.fillStyle = DARK_GREY;
        for (let i = 0; i < dataVals.length; i++) {
            const val = dataVals[i].toFixed(3)
            ctx.fillText(val, xPositions[i] - 30, H_GAUSS + yOffset);
        }
    }

    const drawExtrema = (i0: number = 30, i1: number = CANVAS_WIDTH - 30) => {
        const xData = [i0, i1].map((x) => xPxToData((x), params.start_pf, params.end_pf, CANVAS_WIDTH));
        drawText(xData, [i0, i1], -(10), 24)
        const l = 8
        for (let x of [i0, i1]) {
            drawPoints([x, x], [H_GAUSS - l, H_GAUSS + l], xAxisStyle)
        }

    }

    const redraw = () => {
        if (pfB == null) { return }
        const xData = indices.map((x) => xPxToData((x), params.start_pf, params.end_pf, CANVAS_WIDTH));
        const yData = xData.map((xd) => normalDist(xd, params.mu, params.sigma));
        const maxY = normalDist(params.mu, params.mu, params.sigma);

        const yPoints = yData.map((yd) => yDataToPx(yd, 0, maxY + 0.2, H_GAUSS));
        const sp = getShadedPoints(pfB[0], pfB[1]);

        curveStyle.lineColour = headerHex;
        curveStyle.fillColour = headerHex;
        shadeStyle.lineColour = shadedHex;
        shadeStyle.fillColour = shadedHex;

        clearCanv();
        drawPoints([0, CANVAS_WIDTH], [H_GAUSS, H_GAUSS], xAxisStyle)
        drawExtrema();
        drawPoints([CANVAS_WIDTH / 2, CANVAS_WIDTH / 2], [0, H_GAUSS], yAxisStyle)
        drawPoints(indices, yPoints, curveStyle);
        drawPoints(sp.xPoints, sp.yPoints, shadeStyle);
        drawText([pfB[0], params.mu, pfB[1]!], [sp.xPoints[0], CANVAS_WIDTH / 2, sp.xPoints[sp.xPoints.length - 1]]);
        drawLabels();
    }

    const setConf = (e: any) => {
        setSelectedConf(Number(e.target!.value));
    };

    useEffect(() => {
        // generate, draw
        redraw()
    }, [params])

    useEffect(() => {
        const result = getPredictionInterval(selectedConf / 100, analysisInfo?.pf!, analysisInfo?.cumSumSum!)
        const [lbData, ubData] = [result[0], result[1]]
        const sigma = (ubData - lbData) / 4 // sigma should be fixed as ub and lb changes - this should be reflected in results as well
        console.log(analysisInfo?.stdModel!, sigma)
        const newMaxY = normalDist(phaseFrac, phaseFrac, analysisInfo?.stdModel!)
        const newStartPf = phaseFrac - 4 * sigma
        const newEndPf = phaseFrac + 4 * sigma
        const newParams: NormalParams = { mu: phaseFrac, sigma: sigma, start_pf: newStartPf, end_pf: newEndPf, max_y: newMaxY }
        setParams(newParams)
    }, [analysisInfo])

    useEffect(() => {
        redraw()
    }, [pfB])

    useEffect(() => {
        const result = getPredictionInterval(selectedConf / 100, analysisInfo?.pf!, analysisInfo?.cumSumSum!)
        setPfB(result)
    }, [selectedConf])

    return (
        <div style={{ width: '100%' }}>
            <canvas style={{ width: '70%', marginLeft: '15%', marginBottom: '0.5em' }} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} ref={canvasRef} />
            <InputGroup style={{ width: '70%', marginLeft: '15%' }}> {/*style={{ width: '70%', marginLeft: '15%' }}*/}
                <InputGroup.Text>Confidence in p.f. bounds (%):</InputGroup.Text>

                <Form.Control type="number" min={0} max={100} value={selectedConf} step={0.5} onChange={(e) => setConf(e)} width={1} size="sm"></Form.Control>
                <Form.Range min={80} max={99.999} value={selectedConf} step={0.1} onChange={(e) => setConf(e)} />
            </InputGroup>
        </div>)
}

export default NormalSlider