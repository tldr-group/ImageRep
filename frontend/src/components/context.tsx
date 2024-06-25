import React, { useState } from "react";
import AppContext, { ImageLoadInfo, MenuState } from "./interfaces";

const AppContextProvider = (props: {
    children: React.ReactElement<any, string | React.JSXElementConstructor<any>>;
}) => {

    const [imageInfo, setImageInfo] = useState<ImageLoadInfo | null>(null);
    const [previewImg, setPreviewImg] = useState<HTMLImageElement | null>(null);
    const [selectedPhase, setSelectedPhase] = useState<number>(1);
    const [menuState, setMenuState] = useState<MenuState>('hidden');


    return (
        <AppContext.Provider
            value={{
                imageInfo: [imageInfo, setImageInfo],
                previewImg: [previewImg, setPreviewImg],
                selectedPhase: [selectedPhase, setSelectedPhase],
                menuState: [menuState, setMenuState]
            }}
        >
            {props.children}
        </AppContext.Provider>
    );
}

export default AppContextProvider;