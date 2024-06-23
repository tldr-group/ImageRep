import React, { useState } from "react";
import AppContext, { MenuState } from "./interfaces";

const AppContextProvider = (props: {
    children: React.ReactElement<any, string | React.JSXElementConstructor<any>>;
}) => {

    const [previewData, setPreviewData] = useState<ImageData | null>(null);
    const [previewImg, setPreviewImg] = useState<HTMLImageElement | null>(null);
    const [userFile, setUserFile] = useState<File | null>(null);
    const [nPhases, setNPhases] = useState<number>(2);
    const [selectedPhase, setSelectedPhase] = useState<number>(1);
    const [menuState, setMenuState] = useState<MenuState>('hidden');


    return (
        <AppContext.Provider
            value={{
                previewData: [previewData, setPreviewData],
                previewImg: [previewImg, setPreviewImg],
                userFile: [userFile, setUserFile],
                nPhases: [nPhases, setNPhases],
                selectedPhase: [selectedPhase, setSelectedPhase],
                menuState: [menuState, setMenuState]
            }}
        >
            {props.children}
        </AppContext.Provider>
    );
}

export default AppContextProvider;