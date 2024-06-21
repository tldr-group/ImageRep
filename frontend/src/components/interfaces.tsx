import { createContext } from "react";

interface contextProps {
    image: [
        image: HTMLImageElement | null,
        setImage: (e: HTMLImageElement | null) => void
    ];
    userFile: [
        userFile: File | null,
        setUserFile: (e: File | null) => void
    ];
};


const AppContext = createContext<contextProps | null>(null);
export default AppContext;


export interface DragDropProps {
    loadDefault: () => void;
    loadFromFile: (file: File) => void;
}

export interface imageLoadInfo {
    previewData: ImageData; //use previewData.data to get raw arr
    nDims: 2 | 3;
    nPhases: number;
    segmented: boolean;
    height: number;
    width: number;
}