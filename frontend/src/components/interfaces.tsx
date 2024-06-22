import { createContext } from "react";

interface contextProps {
    previewData: [
        previewData: ImageData | null,
        setPreviewData: (e: ImageData | null) => void
    ];
    previewImg: [
        previewImg: HTMLImageElement | null,
        setPreviewImg: (e: HTMLImageElement | null) => void
    ];
    userFile: [
        userFile: File | null,
        setUserFile: (e: File | null) => void
    ];
    selectedPhase: [
        selectedPhase: number,
        setSelectedPhase: (e: number) => void
    ]

};


const AppContext = createContext<contextProps | null>(null);
export default AppContext;


export interface DragDropProps {
    loadFromFile: (file: File) => void;
}

export interface imageLoadInfo {
    previewData: ImageData; //use previewData.data to get raw arr
    previewImg: HTMLImageElement;
    nDims: 2 | 3;
    nPhases: number;
    segmented: boolean;
    height: number;
    width: number;
}