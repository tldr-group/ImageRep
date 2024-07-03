import { createContext } from "react";

interface contextProps {
    imageInfo: [
        imageInfo: ImageLoadInfo | null,
        setImageInfo: (e: ImageLoadInfo | null) => void
    ]
    previewImg: [
        previewImg: HTMLImageElement | null,
        setPreviewImg: (e: HTMLImageElement | null) => void
    ];
    selectedPhase: [
        selectedPhase: number,
        setSelectedPhase: (e: number) => void
    ];
    selectedConf: [
        selectedConf: number,
        setSelectedNumber: (e: number) => void
    ];
    errVF: [
        errVF: number,
        setErrVF: (e: number) => void,
    ];
    targetL: [
        targetL: number | null,
        setTargetL: (e: number | null) => void,
    ];
    accurateFractions: [
        accurateFractions: { [val: number]: number } | null,
        setAccurateFractions: (e: { [val: number]: number } | null) => void,
    ]
    analysisInfo: [
        analysisInfo: AnalysisInfo | null,
        setAnalysisInfo: (e: AnalysisInfo | null) => void
    ]
    // what do I need to store from azure? cls, z, l
    menuState: [
        menuState: MenuState,
        setMenuState: (e: MenuState) => void
    ];
    errorState: [
        errorState: ErrorMessage,
        setErrorState: (e: ErrorMessage) => void
    ];
    showWarning: [
        showWarning: boolean,
        setShowWarning: (e: boolean) => void
    ]
    showInfo: [
        showInfo: boolean,
        setShowInfo: (e: boolean) => void
    ]
};

const AppContext = createContext<contextProps | null>(null);
export default AppContext;


export interface DragDropProps {
    loadFromFile: (file: File) => void;
}

export interface TopbarProps {
    loadFromFile: (file: File) => void;
    reset: () => void;
}

export function rgbaToHex(r: number, g: number, b: number, a: number) {
    // from user 'Sotos' https://stackoverflow.com/questions/49974145/how-to-convert-rgba-to-hex-color-code-using-javascript
    const red = r.toString(16).padStart(2, '0');
    const green = g.toString(16).padStart(2, '0');
    const blue = b.toString(16).padStart(2, '0');
    const alpha = Math.round(a).toString(16).padStart(2, '0');
    return `#${red}${green}${blue}${alpha}`;
}

export const colours: number[][] = [[255, 255, 255, 255], [31, 119, 180, 255], [255, 127, 14, 255], [44, 160, 44, 255], [214, 39, 40, 255], [148, 103, 189, 255], [140, 86, 75, 255]]
export const IR_LIMIT_PX = 70

export type MenuState = "hidden" | "phase" | "conf" | "processing" | "conf_result" | "length" | "length_result"


export interface ImageLoadInfo {
    file: File | null;
    previewData: ImageData; //use previewData.data to get raw arr
    previewImg: HTMLImageElement;
    nDims: 2 | 3;
    nPhases: number;
    phaseVals: Array<number>;
    segmented: boolean;
    height: number;
    width: number;
    depth: number;
}

export interface AnalysisInfo {
    integralRange: number,
    z: number,
    percentageErr: number,
    absError: number,
    lForDefaultErr: number,
    vf: number
}

export interface ErrorMessage {
    msg: string;
    stackTrace: string;
}

export interface Point {
    x: number;
    y: number
}