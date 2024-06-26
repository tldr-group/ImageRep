import React, { useState } from "react";
import AppContext, { AnalysisInfo, ImageLoadInfo, MenuState } from "./interfaces";

const AppContextProvider = (props: {
    children: React.ReactElement<any, string | React.JSXElementConstructor<any>>;
}) => {
    // user files
    const [imageInfo, setImageInfo] = useState<ImageLoadInfo | null>(null);
    const [previewImg, setPreviewImg] = useState<HTMLImageElement | null>(null);
    // user options
    const [selectedPhase, setSelectedPhase] = useState<number>(0);
    const [selectedConf, setSelectedConf] = useState<number>(95);
    const [errVF, setErrVF] = useState<number>(5);
    const [targetL, setTargetL] = useState<number | null>(null);
    // server data
    const [accurateFractions, setAccurateFractions] = useState<{ [val: number]: number } | null>(null);
    const [analysisInfo, setAnalysisInfo] = useState<AnalysisInfo | null>(null);
    // control flow
    const [menuState, setMenuState] = useState<MenuState>('hidden');


    return (
        <AppContext.Provider
            value={{
                imageInfo: [imageInfo, setImageInfo],
                previewImg: [previewImg, setPreviewImg],
                selectedPhase: [selectedPhase, setSelectedPhase],
                selectedConf: [selectedConf, setSelectedConf],
                errVF: [errVF, setErrVF],
                targetL: [targetL, setTargetL],
                accurateFractions: [accurateFractions, setAccurateFractions],
                analysisInfo: [analysisInfo, setAnalysisInfo],
                menuState: [menuState, setMenuState]
            }}
        >
            {props.children}
        </AppContext.Provider>
    );
}

export default AppContextProvider;