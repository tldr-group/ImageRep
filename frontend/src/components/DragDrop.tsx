import React, { useContext, useEffect, useRef, useState } from "react";
import { DragDropProps } from "./interfaces";

const DEFAULT_IMAGE = "../assets/default.tiff";

export const dragDropStyle = {
    height: '75vh', width: '75vw',
    outline: '10px dashed #b5bab6', color: '#b5bab6',
    fontSize: '3em', justifyContent: 'center', alignItems: 'center',
    borderRadius: '10px', padding: '10px', display: 'flex', margin: 'auto'
}
const flexCenterClasses = "flex items-center justify-center";

const isMobile = (window.innerWidth < 800)

const DragDrop = ({ loadFromFile }: DragDropProps): JSX.Element => {
    // Drag and drop for file upload
    const handleDrag = (e: any) => { e.preventDefault(); }
    const handeDrop = (e: any) => {
        e.preventDefault();
        if (e.dataTransfer.items) {
            const item = e.dataTransfer.items[0];
            if (item.kind === "file") {
                const file = item.getAsFile();
                loadFromFile(file);
            };
        };
    };

    const viewExample = async () => {
        const url = new URL(DEFAULT_IMAGE, location.origin);
        console.log(url)
        const resp = await fetch(url);
        const data = await resp.blob();
        const metadata = { type: data.type }
        const file = new File([data], 'default.tiff', metadata)
        console.log(file)
        loadFromFile(file)
    }

    return (
        <div style={dragDropStyle}
            onDragOver={handleDrag}
            onDrop={handeDrop}
        >
            {(!isMobile) && <span>Drag microstructure file or <a style={{ cursor: "pointer", color: 'blue' }} onClick={viewExample}> view example!</a></span>}
        </div>
    );
}

export default DragDrop