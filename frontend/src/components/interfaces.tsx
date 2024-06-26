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
    menuState: [
        menuState: MenuState,
        setMenuState: (e: MenuState) => void
    ];
};

const AppContext = createContext<contextProps | null>(null);
export default AppContext;


export interface DragDropProps {
    loadFromFile: (file: File) => void;
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