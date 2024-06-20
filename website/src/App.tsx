import React, { useContext, useEffect, useRef, useState } from "react";
import { DragDropProps } from "./components/createContext";


const DragDrop = ({ loadDefault, loadFromFile }: DragDropProps) => {
    // Drag and drop for file upload
    const handleDrag = (e: any) => { e.preventDefault(); }
    const handeDrop = (e: any) => {
        e.preventDefault();
        if (e.dataTransfer.items) {
            const item = e.dataTransfer.items[0]
            if (item.kind === "file") {
                const file = item.getAsFile();
                loadFromFile(file);
            };
        };
    };
    //height: '750px', width: '750px'
    return (
        <div style={{
            height: '80vh', width: '75vw',
            outline: '10px dashed #b5bab6', color: '#b5bab6',
            fontSize: '2em', justifyContent: 'center', alignItems: 'center',
            borderRadius: '10px', padding: '10px'
        }}
            onDragOver={handleDrag}
            onDrop={handeDrop}
        >
            <span>Drag image file(s) or&nbsp; </span> <a style={{ cursor: "pointer", color: 'blue' }} onClick={loadDefault}> view example image</a>
        </div>
    )
}

const App = () => {

    const foo = () => { }
    const fileFoo = (file: File) => { }

    return (<DragDrop loadDefault={foo} loadFromFile={fileFoo} />);
};

export default App;