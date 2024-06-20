import { createContext } from "react";

interface contextProps { };

export interface DragDropProps {
    loadDefault: () => void;
    loadFromFile: (file: File) => void;
}

const AppContext = createContext<contextProps | null>(null);
export default AppContext;